use super::topo_config::{TopoConfig, TopoGraph, TopoMove, TopoMoveStrategy};
use rand::prelude::SliceRandom;
use rand::Rng;

/// Selects the node with the smallest out-degree of a random sample of nodes of the fvs.
/// The sample has size *sample_size*.
///
/// Uses the i-(v) and i+(v) functions, which are presented in the Galinier et al. paper, to
/// determine the position in the topological sorting.
pub struct RandomTopoStrategy<'a, R> {
    rng: &'a mut R,

    /// Amount of nodes that will be evaluated as move candidates in every next_move call.
    sample_size: usize,
}

impl<'a, R> RandomTopoStrategy<'a, R>
where
    R: Rng,
{
    pub fn new(rng: &'a mut R, sample_size: usize) -> RandomTopoStrategy<'a, R> {
        Self { rng, sample_size }
    }
}

impl<'a, G, R> TopoMoveStrategy<G> for RandomTopoStrategy<'a, R>
where
    G: TopoGraph,
    R: Rng,
{
    fn next_move<'b, T>(&mut self, topo_config: &T) -> Option<TopoMove>
    where
        T: TopoConfig<'b, G>,
    {
        if topo_config.fvs().is_empty() {
            return None;
        }

        let random_node = (0..self.sample_size)
            .map(|_| *topo_config.fvs().choose(&mut self.rng).unwrap())
            .min_by_key(|node| topo_config.graph().total_degree(*node))
            .unwrap();

        let random_position = topo_config.get_i_index(random_node);

        Some(topo_config.create_move(random_node, random_position))
    }
}

#[cfg(test)]
mod test {
    use super::super::topo_local_search::TopoLocalSearch;
    use super::super::vec_topo_config::VecTopoConfig;
    use super::*;
    use crate::algorithm::TerminatingIterativeAlgorithm;
    use crate::bench::fvs_bench::test_utils::test_algo_with_pace_graphs;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::Traversal;
    use crate::heuristics::local_search::sim_anneal::SimAnneal;
    use crate::heuristics::utils::apply_fvs_to_graph;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn test_simple_cyclic_graph() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            RandomTopoStrategy::new(&mut strategy_rng, 7),
        );
        let mut sim_anneal = SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
        let fvs = sim_anneal.run_to_completion().unwrap();
        assert_eq!(fvs.len(), 1);

        let reduced_graph = apply_fvs_to_graph(&graph, fvs);
        assert!(reduced_graph.is_acyclic());
    }

    #[test]
    fn test_with_graphs() {
        test_algo_with_pace_graphs("RandomTopoStrategy", |graph, _, _, _| {
            let mut strategy_rng = Pcg64::seed_from_u64(0);
            let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
            let topo_config = VecTopoConfig::new(&graph);
            let strategy = RandomTopoStrategy::new(&mut strategy_rng, 7);
            let local_search = TopoLocalSearch::new(topo_config, strategy);
            let mut sim_anneal =
                SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
            sim_anneal.run_to_completion().unwrap()
        })
        .unwrap();
    }
}
