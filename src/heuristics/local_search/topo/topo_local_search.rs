use super::topo_config::{TopoConfig, TopoGraph, TopoMove, TopoMoveStrategy};
use crate::graph::Node;
use std::marker::PhantomData;

/// Wraps a [`TopoConfig`] and a [`TopoMoveStrategy`] and exposes methods to use these for a local
/// search.
pub struct TopoLocalSearch<T, S, G> {
    topo_config: Option<T>,
    strategy: Option<S>,
    _phantom_graph: PhantomData<G>,
}

impl<'a, T, S, G> TopoLocalSearch<T, S, G>
where
    G: TopoGraph,
    T: TopoConfig<'a, G> + 'a,
    S: TopoMoveStrategy<G>,
{
    pub fn new(topo_config: T, strategy: S) -> Self {
        Self {
            topo_config: Some(topo_config),
            strategy: Some(strategy),
            _phantom_graph: Default::default(),
        }
    }

    /// Retrieves the next move proposed by the wrapped [TopoMoveStrategy]
    pub fn next_move(&mut self) -> Option<TopoMove> {
        let topo_config = self
            .topo_config
            .take()
            .expect("next_move called recursively!");
        let mut strategy = self.strategy.take().expect("next_move called recursively!");

        let topo_move = strategy.next_move(&topo_config);

        self.topo_config = Some(topo_config);
        self.strategy = Some(strategy);

        topo_move
    }

    pub fn perform_move(&mut self, mut topo_move: TopoMove) {
        let mut topo_config = self
            .topo_config
            .take()
            .expect("perform_move called recursively!");
        let mut strategy = self
            .strategy
            .take()
            .expect("perform_move called recursively!");

        strategy.on_before_perform_move(&topo_config, &mut topo_move);
        topo_config.perform_move(topo_move);

        self.topo_config = Some(topo_config);
        self.strategy = Some(strategy);
    }

    pub fn reject_move(&mut self, topo_move: TopoMove) {
        let topo_config = self
            .topo_config
            .take()
            .expect("reject_move called recursively!");
        let mut strategy = self
            .strategy
            .take()
            .expect("reject_move called recursively!");

        strategy.on_move_rejected(&topo_config, &topo_move);

        self.topo_config = Some(topo_config);
        self.strategy = Some(strategy);
    }

    pub fn fvs(&self) -> &[Node] {
        self.topo_config.as_ref().unwrap().fvs()
    }
}
