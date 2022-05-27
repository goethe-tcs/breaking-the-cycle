#![deny(warnings)]

use dfvs::graph::*;
use fxhash::FxHashSet;
use itertools::Itertools;
use log::*;
use std::collections::VecDeque;
use std::io::stdin;
use std::time::{Duration, Instant};

use dfvs::bitset::BitSet;
use dfvs::graph::io::MetisRead;

type Graph = AdjArrayUndir;

use dfvs::algorithm::IterativeAlgorithm;
use dfvs::exact::BranchAndBound;

use dfvs::kernelization::*;

use dfvs::heuristics::local_search::sim_anneal::SimAnneal;
use dfvs::heuristics::local_search::topo::candidate_topo_strategy::CandidateTopoStrategy;
use dfvs::heuristics::local_search::topo::topo_config::TopoConfig;
use dfvs::heuristics::local_search::topo::topo_local_search::TopoLocalSearch;
use dfvs::heuristics::local_search::topo::vec_topo_config::VecTopoConfig;
use dfvs::heuristics::weakest_link::weakest_link;
use dfvs::log::build_pace_logger_for_level;
use dfvs::signal_handling;
#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> std::io::Result<()> {
    let solver_start = Instant::now();
    build_pace_logger_for_level(LevelFilter::Off);
    signal_handling::initialize();

    let graph = {
        let stdin = stdin();
        Graph::try_read_metis(stdin.lock())?
    };

    info!(
        "Input graph with n={}, m={}",
        graph.number_of_nodes(),
        graph.number_of_edges()
    );

    // reduce graph
    let (red_fvs, sccs) = {
        let mut super_reducer = SuperReducer::with_settings(
            graph.clone(),
            vec![
                Rules::Rule1,
                Rules::STOP,
                Rules::Rule3,
                Rules::Rule4,
                Rules::PIE,
                Rules::RestartRules,
                Rules::DiClique,
                Rules::C4,
                Rules::RestartRules,
                Rules::DOME,
                Rules::Rule5(2000),
                Rules::Crown(2000),
                Rules::CompleteNode,
            ],
            true,
        );
        let (red_fvs, sccs) = super_reducer.reduce().unwrap();
        let red_fvs = red_fvs.clone();
        let mut sccs = sccs.clone();

        sccs.sort_by_key(|(graph, _)| graph.len());

        trace!("Initial Reduction completed");
        (red_fvs, sccs)
    };

    // TODO: Experiment with heuristic 'reduction rules'

    // process sccs
    let mut solution = if sccs.is_empty() {
        red_fvs
    } else {
        process_sccs(red_fvs, sccs, solver_start)?
    };
    solution.sort_unstable();
    info!("Solution has size {}", solution.len());

    // Verify solution
    {
        info!("Verify solution {}", solution.len());
        let mask = BitSet::new_all_set_but(graph.len(), solution.clone());
        let sol = graph.vertex_induced(&mask).0;
        assert!(sol.is_acyclic());
        assert_eq!(graph.len() - mask.cardinality(), solution.len());
    }

    // Output solution
    println!(
        "{}",
        solution.iter().map(|&x| format!("{}", x + 1)).join("\n")
    );

    Ok(())
}
fn process_sccs(
    mut solution: Vec<Node>,
    sccs: Vec<(Graph, NodeMapper)>,
    solver_start: Instant,
) -> std::io::Result<Vec<Node>> {
    info!("{} SCCs", sccs.len());

    info!("Trying exact algo...");
    let time_for_exact = Duration::from_secs(60);
    let exact_time_resolution =
        Duration::from_secs_f64(time_for_exact.as_secs() as f64 / sccs.len() as f64);
    let mut solved_sccs = FxHashSet::with_capacity_and_hasher(sccs.len(), Default::default());
    let mut queue: VecDeque<_> = sccs
        .iter()
        .enumerate()
        .filter(|(_id, (scc, _mapper))| scc.number_of_edges() <= 20000)
        .map(|(id, (scc, mapper))| (id, (BranchAndBound::new(scc.clone()), mapper.clone())))
        .collect();
    let start_exact = Instant::now();
    while start_exact.elapsed() < time_for_exact {
        let next_scc = queue.pop_front();
        if next_scc.is_none() {
            info!("Exact queue was empty before allocated time for exact algo was reached!");
            break;
        }

        if signal_handling::received_ctrl_c() {
            info!("No time for exact solver: Received ctrl c! Aborting...");
            break;
        }

        let (id, (mut bnb, mapper)) = next_scc.unwrap();

        bnb.run_until_timeout(exact_time_resolution);
        if bnb.is_completed() {
            trace!("Solved SCC using exact algo!");
            let scc_fvs = bnb.best_known_solution().unwrap().to_vec();
            solution.extend(mapper.get_old_ids(scc_fvs.into_iter()));
            solved_sccs.insert(id);
        } else {
            // push scc back in queue with an allowed time of one second, in case some sccs are
            // solved faster than their allocated time and we thus have time left for sccs that were
            // not solved in their allocated time
            queue.push_back((id, (bnb, mapper)));
        }
    }

    let unsolved_sccs = sccs
        .into_iter()
        .enumerate()
        .filter_map(|(id, (scc, mapper))| {
            if solved_sccs.contains(&id) {
                None
            } else {
                Some((scc, mapper))
            }
        })
        .enumerate()
        .collect_vec();
    let mut time_fac = 1.0
        / ((unsolved_sccs
            .iter()
            .map(|(_id, (scc, _))| (scc.number_of_edges() as u64).pow(2)))
        .sum::<u64>() as f64);
    time_fac *= (Duration::from_secs(600) - solver_start.elapsed()).as_secs_f64();
    info!(
        "Solving remaining {} SCCs using heuristic approach...",
        unsolved_sccs.len()
    );
    let scc_amount = unsolved_sccs.len();
    for (id, (scc, mapper)) in unsolved_sccs {
        let scc_start = Instant::now();
        let time =
            Duration::from_secs_f64(((scc.number_of_edges() as u64).pow(2) as f64) * time_fac);

        trace!("Running Weakest Link...");
        let weakest_start = Instant::now();
        let max_weakest_runtime = time;
        let weakest_link_fvs = weakest_link(scc.clone(), || {
            weakest_start.elapsed() > max_weakest_runtime || signal_handling::received_ctrl_c()
        });

        // no time left: include all nodes of scc
        if signal_handling::received_ctrl_c() {
            info!("No time for SCC left, appending all its nodes to the DFVS!");
            let scc_fvs = scc.vertices().collect_vec();
            solution.extend(mapper.get_old_ids(scc_fvs.into_iter()));
            continue;
        }

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let topo_config = if let Some(weakest_link_fvs) = weakest_link_fvs {
            trace!("Weakest Link finished...");
            VecTopoConfig::new_with_fvs(&scc, weakest_link_fvs)
        } else {
            trace!("Weakest Link didn't finish");
            VecTopoConfig::new(&scc)
        };

        trace!("Running Simulated Annealing...");
        let strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);
        let local_search = TopoLocalSearch::new(topo_config, strategy);

        // run until time limit if this is the last SCC
        let (max_anneal_runtime, max_fails) = if id == scc_amount - 1 {
            trace!("Processing last SCC until SIGINT is received...");
            (Duration::from_secs(600), 99999999999)
        } else {
            (time - scc_start.elapsed(), 120)
        };
        let n = scc.number_of_nodes() as f64;
        let max_evals = (n * 0.1 * n.log10()) as usize;
        let mut sim_anneal = SimAnneal::new(
            local_search,
            max_evals,
            max_fails,
            0.25,
            0.999,
            &mut sim_anneal_rng,
        );
        sim_anneal.run_until_timeout(max_anneal_runtime);
        let scc_fvs = sim_anneal.best_known_solution().unwrap();
        solution.extend(mapper.get_old_ids(scc_fvs.iter().copied()));
    }

    Ok(solution)
}
