use super::topo_config::{TopoConfig, TopoGraph, TopoMove, TopoMoveStrategy};
use crate::graph::Node;
use crate::heuristics::local_search::topo::topo_config::MovePosition;
use rand::prelude::SliceRandom;
use rand::Rng;

/// The move strategy that is described in the "Applying local search to the feedback vertex set
/// problem" paper by Galinier et al.
///
/// It caches the i- and i+ performance for every node. The performance values are calculated
/// lazily.
pub struct CandidateTopoStrategy<'a, R> {
    rng: &'a mut R,
    is_dirty: Vec<bool>,

    /// The i- performance of all nodes
    deltas_minus: Vec<isize>,

    /// The i+ performance of all nodes
    deltas_plus: Vec<isize>,
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
            deltas_plus: vec![0; topo_config.graph().len()],
            deltas_minus: vec![0; topo_config.graph().len()],
        }
    }

    fn recalc_node_performances<'b, G, T>(
        &mut self,
        topo_config: &T,
        node: Node,
        use_i_minus: bool,
    ) -> TopoMove
    where
        G: TopoGraph,
        T: TopoConfig<'b, G>,
    {
        let (i_minus_move, i_plus_move) = topo_config.calc_move_candidates(node);
        self.deltas_minus[node as usize] = i_minus_move.performance();
        self.deltas_plus[node as usize] = i_plus_move.performance();
        self.is_dirty[node as usize] = false;

        if use_i_minus {
            i_minus_move
        } else {
            i_plus_move
        }
    }

    fn get_cached_move(&mut self, node: Node, use_i_minus: bool) -> TopoMove {
        if use_i_minus {
            TopoMove::new_as_cached(node, MovePosition::IMinus, self.deltas_minus[node as usize])
        } else {
            TopoMove::new_as_cached(node, MovePosition::IPlus, self.deltas_plus[node as usize])
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
            let use_i_minus = self.rng.gen_bool(0.5);

            if self.is_dirty[node as usize] {
                Some(self.recalc_node_performances(topo_config, node, use_i_minus))
            } else {
                // Some(self.recalc_node_performances(topo_config, node, use_i_minus))
                Some(self.get_cached_move(node, use_i_minus))
            }
        }
        // no move possible because fvs is empty
        else {
            None
        }
    }

    fn on_before_perform_move<'b, T>(&mut self, topo_config: &T, topo_move: &mut TopoMove)
    where
        T: TopoConfig<'b, G>,
    {
        let node = topo_move.node();
        let dirty_nodes = topo_move
            .get_or_calc_conflicts(topo_config)
            .iter()
            .map(|(node, _index)| *node)
            .chain(topo_config.graph().out_neighbors(node))
            .chain(topo_config.graph().in_neighbors(node));

        for dirty_node in dirty_nodes {
            self.is_dirty[dirty_node as usize] = true;
        }
        self.is_dirty[topo_move.node() as usize] = true;
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
    use crate::graph::{GraphOrder, Traversal};
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

    #[test]
    fn test_with_graphs() {
        test_algo_with_pace_graphs("CandidateTopoStrategy", |graph, _, _, _| {
            let mut strategy_rng = Pcg64::seed_from_u64(0);
            let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
            let topo_config = VecTopoConfig::new(&graph);
            let strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);
            let local_search = TopoLocalSearch::new(topo_config, strategy);
            let mut sim_anneal =
                SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
            sim_anneal.run_to_completion().unwrap()
        })
        .unwrap();
    }

    #[test]
    fn on_before_perform_move_sanity_check() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);
        let mut strategy_rng = Pcg64::seed_from_u64(0);

        let mut topo_config = VecTopoConfig::new(&graph);
        topo_config.set_state_from_fvs([0, 1]);
        let mut strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);

        for _ in 0..10 {
            let mut next_move = strategy.next_move(&topo_config).unwrap();
            let node = next_move.node() as usize;

            assert!(!strategy.is_dirty[node]);
            strategy.on_before_perform_move(&topo_config, &mut next_move);
            assert!(strategy.is_dirty[node]);

            topo_config.perform_move(next_move);
        }
    }

    #[test]
    fn test_on_before_perform_move() {
        let graph = AdjArrayIn::from(&[
            (0, 1),
            (2, 0),
            (0, 3),
            (4, 0),
            (5, 0),
            (0, 6),
            (0, 7),
            (9, 9),
        ]);
        let topo_order = vec![1, 2, 9, 3, 4, 5, 6, 8, 7];
        let fvs = vec![0];
        let mut topo_config = VecTopoConfig::new(&graph);
        topo_config.set_state(topo_order.clone(), fvs.clone());
        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut strategy = CandidateTopoStrategy::new(&mut strategy_rng, &topo_config);

        let (mut i_minus_move, mut i_plus_move) = topo_config.calc_move_candidates(0);
        let expected = vec![true, true, true, true, true, true, true, true, false, false];

        strategy.is_dirty = vec![false; topo_config.graph().len()];
        strategy.on_before_perform_move(&topo_config, &mut i_minus_move);
        assert_eq!(strategy.is_dirty, expected);

        strategy.is_dirty = vec![false; topo_config.graph().len()];
        strategy.on_before_perform_move(&topo_config, &mut i_plus_move);
        assert_eq!(strategy.is_dirty, expected);
    }
}
