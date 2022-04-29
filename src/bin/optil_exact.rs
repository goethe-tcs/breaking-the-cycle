#![deny(warnings)]
use dfvs::graph::*;
use itertools::Itertools;
use log::*;
use std::io::stdin;

use dfvs::bitset::BitSet;
use dfvs::graph::io::MetisRead;

type Graph = AdjArrayUndir;

use dfvs::algorithm::TerminatingIterativeAlgorithm;
use dfvs::exact::BranchAndBound;

use dfvs::kernelization::*;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> std::io::Result<()> {
    let graph = {
        let stdin = stdin();
        Graph::try_read_metis(stdin.lock())?
    };

    info!(
        "Input graph with n={}, m={}",
        graph.number_of_nodes(),
        graph.number_of_edges()
    );

    let mut super_reducer = SuperReducer::with_optimal_pace_settings(graph.clone());
    let (solution, sccs) = super_reducer.reduce().unwrap();
    let mut solution = solution.clone();
    let mut sccs = sccs.clone();
    sccs.sort_by_key(|(g, _)| g.len());

    for (graph, mapper) in sccs {
        let scc_solution = {
            let mut algo = BranchAndBound::new(graph.clone());
            algo.run_to_completion()
                .unwrap()
                .iter()
                .copied()
                .collect_vec()
        };

        solution.extend(scc_solution.iter().map(|&x| mapper.old_id_of(x).unwrap()));
        info!(
            "Processed SCC with n={:>5} and m={:>7}. Solution cardinality {:>5}.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            scc_solution.len(),
        );
    }

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
