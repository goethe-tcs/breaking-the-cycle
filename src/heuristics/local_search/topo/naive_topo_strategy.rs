use super::topo_config::{TopoConfig, TopoGraph, TopoMove, TopoMoveStrategy};
use rand::prelude::*;
use rand::Rng;

/// Selects a move by choosing a random node of the DFVS and a random index in the topological
/// sorting.
pub struct NaiveTopoStrategy<'a, R> {
    rng: &'a mut R,
}

impl<'a, R: Rng> NaiveTopoStrategy<'a, R> {
    pub fn new(rng: &'a mut R) -> NaiveTopoStrategy<'a, R> {
        Self { rng }
    }
}

impl<'a, G, R> TopoMoveStrategy<G> for NaiveTopoStrategy<'a, R>
where
    G: TopoGraph,
    R: Rng,
{
    fn next_move<'b, T>(&mut self, topo_config: &T) -> Option<TopoMove>
    where
        T: TopoConfig<'b, G>,
    {
        if let Some(&random_node) = topo_config.fvs().choose(self.rng) {
            let indices = 0..topo_config.len();
            let random_position = indices.choose(&mut self.rng).unwrap_or(0);

            Some(topo_config.create_move(random_node, random_position))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests_rand_strategy {
    use super::super::topo_local_search::TopoLocalSearch;
    use super::super::vec_topo_config::VecTopoConfig;
    use super::*;
    use crate::algorithm::TerminatingIterativeAlgorithm;
    use crate::bench::fvs_bench::test_utils::test_algo_with_pace_graphs;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::Traversal;
    use crate::heuristics::local_search::sim_anneal::SimAnneal;
    use crate::heuristics::utils::apply_fvs_to_graph;
    use rand_pcg::Pcg64;

    #[test]
    fn test_simple_cyclic_graph() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            NaiveTopoStrategy::new(&mut strategy_rng),
        );
        let mut sim_anneal = SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
        let fvs = sim_anneal.run_to_completion().unwrap();
        assert_eq!(fvs.len(), 1);

        let reduced_graph = apply_fvs_to_graph(&graph, fvs);
        assert!(reduced_graph.is_acyclic());
    }

    #[test]
    fn test_with_graphs() {
        test_algo_with_pace_graphs("NaiveTopoStrategy", |graph, _, _, _| {
            let mut strategy_rng = Pcg64::seed_from_u64(0);
            let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
            let topo_config = VecTopoConfig::new(&graph);
            let strategy = NaiveTopoStrategy::new(&mut strategy_rng);
            let local_search = TopoLocalSearch::new(topo_config, strategy);
            let mut sim_anneal =
                SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
            sim_anneal.run_to_completion().unwrap()
        })
        .unwrap();
    }
}
