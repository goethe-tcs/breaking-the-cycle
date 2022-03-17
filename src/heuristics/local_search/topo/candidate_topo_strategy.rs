use super::topo_config::{TopoConfig, TopoGraph, TopoMove, TopoMoveStrategy};
use crate::graph::Node;
use rand::prelude::SliceRandom;
use rand::Rng;

/// The move strategy that is described in the "Applying local search to the feedback vertex set
/// problem" paper by Galinier et al.
///
/// It caches the i- and i+ indices for every node as move candidates. The indices are calculated
/// lazily.
pub struct CandidateTopoStrategy<'a, R> {
    rng: &'a mut R,
    is_dirty: Vec<bool>,

    /// The i- index candidates of all nodes
    indices_minus: Vec<Node>,

    /// The i+ index candidates of all nodes
    indices_plus: Vec<Node>,
}

impl<'a, R: Rng> CandidateTopoStrategy<'a, R> {
    pub fn new<'b, G, T>(rng: &'a mut R, topo_config: &T) -> CandidateTopoStrategy<'a, R>
    where
        G: TopoGraph,
        T: TopoConfig<'b, G>,
    {
        Self {
            rng,
            is_dirty: vec![true; topo_config.graph().len()],
            indices_plus: vec![0; topo_config.graph().len()],
            indices_minus: vec![0; topo_config.graph().len()],
        }
    }
}

impl<'a, G, R> TopoMoveStrategy<G> for CandidateTopoStrategy<'a, R>
where
    G: TopoGraph,
    R: Rng,
{
    fn next_move<'b, T>(&mut self, topo_config: &T) -> Option<TopoMove>
    where
        T: TopoConfig<'b, G>,
    {
        if let Some(&node) = topo_config.fvs().choose(self.rng) {
            if self.is_dirty[node as usize] {
                self.indices_minus[node as usize] = topo_config.get_i_minus_index(node).0 as Node;
                self.indices_plus[node as usize] = topo_config.get_i_plus_index(node).0 as Node;
                self.is_dirty[node as usize] = false;
            }

            let index = if self.rng.gen_bool(0.5) {
                self.indices_plus[node as usize]
            } else {
                self.indices_minus[node as usize]
            };

            Some(topo_config.create_move(node, index as usize))
        } else {
            None
        }
    }

    fn on_before_perform_move<'b, T>(&mut self, _topo_config: &T, topo_move: &TopoMove)
    where
        T: TopoConfig<'b, G>,
    {
        for (dirty_node, _) in &topo_move.incompatible_neighbors {
            self.is_dirty[*dirty_node as usize] = true;
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::topo_local_search::TopoLocalSearch;
    use super::super::vec_topo_config::VecTopoConfig;
    use super::*;
    use crate::algorithm::TerminatingIterativeAlgorithm;
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
        let topo_config = VecTopoConfig::new(&graph);
        let strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);
        let local_search = TopoLocalSearch::new(topo_config, strategy);
        let mut sim_anneal = SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
        let fvs = sim_anneal.run_to_completion().unwrap();

        let reduced_graph = apply_fvs_to_graph(&graph, fvs.clone());
        assert!(reduced_graph.is_acyclic());
        assert_eq!(fvs.len(), 1);
    }
}
