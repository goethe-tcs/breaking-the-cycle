use super::{TopoMoveStrategy, TopologicalConfig};
use crate::graph::{AdjacencyListIn, GraphEdgeEditing, GraphOrder, Node};
use rand::Rng;

/// Implementation of the simulated annealing algorithm that is presented in the
/// "Applying local search to the feedback vertex set problem" paper by Philippe Galinier et al.
/// Assumes the graph has no self-loops
pub fn sim_anneal<
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    M: TopoMoveStrategy<G>,
    R: Rng,
>(
    graph: &G,
    move_strategy: &mut M,
    max_stage_evals: usize,
    max_stage_fails: usize,
    start_temperature: f64,
    temp_reduce_fac: f64,
    rng: &mut R,
) -> Vec<Node> {
    let mut topo_config = TopologicalConfig::new(graph);
    let mut best_fvs = topo_config.get_fvs();
    let mut curr_temperature = start_temperature;
    let mut failed_stages = 0;

    // run until max_fails many stages failed in succession
    while failed_stages < max_stage_fails {
        // perform one stage
        let mut failure = true;
        for _ in 0..max_stage_evals {
            // get and evaluate move
            let next_move = move_strategy.next_move(&topo_config);
            let delta = next_move.incompatible_neighbors.len() as isize - 1;

            // decide whether to execute the move
            if delta <= 0 || rng.gen_bool((-delta as f64 / curr_temperature).exp()) {
                topo_config.perform_move(next_move);
                if topo_config.not_in_s.len() < best_fvs.len() {
                    best_fvs = topo_config.get_fvs();
                    failure = false;
                }
            }
        }

        // evaluate stage progress
        if failure {
            failed_stages += 1;
        } else {
            failed_stages = 0;
        }
        curr_temperature *= temp_reduce_fac;
    }

    best_fvs
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::Traversal;
    use crate::heuristics::local_search::rand_topo_strategy::RandomTopoStrategy;
    use crate::heuristics::utils::apply_fvs_to_graph;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn test_simple_cyclic_graph() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let mut move_strategy = RandomTopoStrategy::new(&mut strategy_rng, 7);
        let fvs = sim_anneal(
            &graph,
            &mut move_strategy,
            20,
            20,
            1.0,
            0.9,
            &mut sim_anneal_rng,
        );
        assert_eq!(fvs.len(), 1);

        let reduced_graph = apply_fvs_to_graph(&graph, fvs);
        assert!(reduced_graph.is_acyclic());
    }
}
