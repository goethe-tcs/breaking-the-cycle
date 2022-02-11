use super::{TopoMove, TopoMoveStrategy, TopologicalConfig};
use crate::graph::{AdjacencyListIn, GraphEdgeEditing, GraphOrder, Node};
use rand::Rng;

/// Selects the node with the smallest out-degree of a random sample of nodes
/// which are not yet in S. The sample has size *sample_size*.
///
/// Uses the i_(v) and i+(v) functions, which are presented in the Galinier et al. paper, to
/// determine the position in S.
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

    /// Get the left most index-candidate in S without creating a conflict with in-neighbors
    /// which are already in S
    fn get_pos_i_minus<'b, G>(
        &self,
        topo_config: &'b TopologicalConfig<G>,
        node: Node,
    ) -> (usize, Vec<usize>)
    where
        G: crate::graph::AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    {
        let neighbor_indices: Vec<usize> = topo_config
            .graph
            .in_neighbors(node)
            .filter_map(|u| topo_config.get_index_of_node(u))
            .collect();

        let pos_i_minus = neighbor_indices.iter().max().map_or(0, |x| x + 1);

        (pos_i_minus, neighbor_indices)
    }

    /// Get the right most index-candidate in S without creating a conflict with out-neighbors
    /// which are already in S
    fn get_pos_i_plus<'b, G>(
        &self,
        topo_config: &'b TopologicalConfig<G>,
        node: Node,
    ) -> (usize, Vec<usize>)
    where
        G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    {
        let neighbor_indices: Vec<usize> = topo_config
            .graph
            .out_neighbors(node)
            .filter_map(|u| topo_config.get_index_of_node(u))
            .collect();

        let pos_i_plus = neighbor_indices
            .iter()
            .min()
            .map_or(topo_config.s.len(), |x| *x);

        (pos_i_plus, neighbor_indices)
    }

    fn get_optimal_s_index<'b, G>(&self, topo_config: &'b TopologicalConfig<G>, node: Node) -> usize
    where
        G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    {
        let (pos_i_minus, in_neighbour) = self.get_pos_i_minus(topo_config, node);
        let (pos_i_plus, out_neighbour) = self.get_pos_i_plus(topo_config, node);
        if in_neighbour > out_neighbour {
            pos_i_minus
        } else {
            pos_i_plus
        }
    }
}

impl<'a, G, R> TopoMoveStrategy<G> for RandomTopoStrategy<'a, R>
where
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    R: Rng,
{
    fn next_move<'b>(&mut self, topo_config: &'b TopologicalConfig<G>) -> TopoMove {
        debug_assert!(!topo_config.not_in_s.is_empty());

        let random_node = (0..self.sample_size)
            .map(|_| *topo_config.not_in_s.choose(&mut self.rng).unwrap())
            .min_by_key(|node| topo_config.graph.total_degree(*node))
            .unwrap();

        let random_position = self.get_optimal_s_index(topo_config, random_node);

        topo_config.create_move(random_node, random_position)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::Traversal;
    use crate::heuristics::local_search::sim_anneal::sim_anneal;
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
