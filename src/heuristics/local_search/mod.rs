use self::topo_config::{TopoMove, TopologicalConfig};

pub mod naive_topo_strategy;
pub mod rand_topo_strategy;
pub mod sim_anneal;
pub mod topo_config;

/// Encapsulates the functionality of deciding about which vertex to add to a topological
/// configuration and at which position.
pub trait TopoMoveStrategy<G> {
    /// Returns which node should be added to the topological configuration and at which position
    fn next_move<'a>(&mut self, topo_config: &'a TopologicalConfig<G>) -> TopoMove;
}
