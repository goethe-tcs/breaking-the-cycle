use super::{TopoMove, TopoMoveStrategy, TopologicalConfig};
use crate::graph::{AdjacencyListIn, GraphEdgeEditing, GraphOrder};
use rand::prelude::*;
use rand::Rng;

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
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    R: Rng,
{
    fn next_move<'b>(&mut self, topo_config: &'b TopologicalConfig<G>) -> TopoMove {
        let random_node = *topo_config
            .not_in_s
            .choose(self.rng)
            .expect("Can't pick next move: All nodes are in S!");

        let indices = 0..topo_config.s.len();
        let random_position = indices.choose(&mut self.rng).unwrap_or(0);

        topo_config.create_move(random_node, random_position)
    }
}

#[cfg(test)]
mod tests_rand_strategy {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::Traversal;
    use crate::heuristics::local_search::sim_anneal::sim_anneal;
    use crate::heuristics::utils::apply_fvs_to_graph;
    use rand_pcg::Pcg64;

    #[test]
    fn test_simple_cyclic_graph() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let mut move_strategy = NaiveTopoStrategy::new(&mut strategy_rng);
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
