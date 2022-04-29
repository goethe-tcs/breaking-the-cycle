#![deny(warnings)]
use dfvs::graph::*;
use itertools::Itertools;
use log::*;
use std::io::stdin;
use std::time::Duration;

use dfvs::bitset::BitSet;
use dfvs::graph::io::MetisRead;

type Graph = AdjArrayUndir;

use dfvs::algorithm::{IterativeAlgorithm, TerminatingIterativeAlgorithm};
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
        let mut super_reducer = SuperReducer::with_optimal_pace_settings(graph.clone());
        let (red_fvs, sccs) = super_reducer.reduce().unwrap();
        let red_fvs = red_fvs.clone();
        let mut sccs = sccs.clone();
        sccs.sort_by_key(|(graph, _)| graph.len());

        trace!("Initial Reduction completed");
        (red_fvs, sccs)
    };

    // TODO: Experiment with heuristic 'reduction rules'

    // process sccs
    let mut solution = process_sccs(red_fvs, sccs)?;
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
) -> std::io::Result<Vec<Node>> {
    let scc_amount = sccs.len();
    trace!("Sccs: {}", scc_amount);
    // TODO: Process multiple sccs at the same time, instead of one after another?
    for (id, (scc, mapper)) in sccs.into_iter().enumerate() {
        info!(
            "Processing SCC with n={} and m={}.",
            scc.number_of_nodes(),
            scc.number_of_edges()
        );

        // no time left: include all nodes of scc
        if signal_handling::received_ctrl_c() {
            info!("No time for SCC left, appending all its nodes to the DFVS!");
            let scc_fvs = scc.vertices().collect_vec();
            solution.extend(mapper.get_old_ids(scc_fvs.into_iter()));
            continue;
        }

        // try exact solver for a limited time
        // TODO: Find out the max scc size that doesn't cause a stack overflow
        if scc.len() <= 10000 {
            trace!("Trying exact algo...");
            let mut algo = BranchAndBound::new(scc.clone());
            // TODO: Experiment with heuristic that determines timeout for exact algo based on SCCs
            algo.run_until_timeout(Duration::from_secs(5));
            if algo.is_completed() {
                trace!("Solved SCC using exact algo!");
                let scc_fvs = algo.best_known_solution().unwrap().to_vec();
                solution.extend(mapper.get_old_ids(scc_fvs.into_iter()));
                continue;
            }
        }

        trace!("Solving SCC using heuristic approach...");
        // TODO: Add time limit to weakest link
        let weakest_link_fvs = weakest_link(scc.clone());

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let topo_config = VecTopoConfig::new_with_fvs(&scc, weakest_link_fvs);
        let strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);
        let local_search = TopoLocalSearch::new(topo_config, strategy);

        // run until time limit if this is the last SCC
        let max_fails = if id == scc_amount - 1 {
            trace!("Processing last SCC until SIGINT is received...");
            9999999999
        } else {
            120
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
        let scc_fvs = sim_anneal.run_to_completion().unwrap();
        solution.extend(mapper.get_old_ids(scc_fvs.into_iter()));
    }

    Ok(solution)
}
