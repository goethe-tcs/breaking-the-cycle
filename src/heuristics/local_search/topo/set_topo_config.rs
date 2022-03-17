use super::macros::impl_helper_topo_new_with_fvs;
use super::topo_config::{TopoConfig, TopoGraph, TopoMove};
use crate::graph::Node;
use crate::heuristics::utils::node_index_set::NodeIndexSet;
use crate::heuristics::utils::set_vec::HashSetVec;
use std::iter::FromIterator;

/// Wraps a [`NodeIndexSet`] for the topological sorting to achieve fast index lookup.
#[derive(Clone)]
pub struct SetTopoConfig<'a, G> {
    graph: &'a G,
    topo_order: NodeIndexSet,
    fvs: HashSetVec<Node>,
}

impl<'a, G> SetTopoConfig<'a, G>
where
    G: TopoGraph,
{
    pub fn new(graph: &'a G) -> Self {
        Self {
            topo_order: NodeIndexSet::new(graph.len()),
            fvs: HashSetVec::from_graph(graph),
            graph,
        }
    }

    impl_helper_topo_new_with_fvs!();
}

impl<'a, G> TopoConfig<'a, G> for SetTopoConfig<'a, G>
where
    G: TopoGraph,
{
    fn set_state<I>(&mut self, topo_order: I, fvs: I)
    where
        I: IntoIterator<Item = Node>,
    {
        let mut new_topo_set = NodeIndexSet::new(self.graph.len());
        new_topo_set.extend(topo_order);
        self.topo_order = new_topo_set;

        self.fvs = HashSetVec::from_iter(fvs.into_iter());
    }

    fn perform_move(&mut self, topo_move: TopoMove) {
        debug_assert!(self.fvs().contains(&topo_move.node));

        let node = topo_move.node;
        let mut position = topo_move.position;
        let mut incompatible_neighbors = topo_move.incompatible_neighbors;

        // sort neighbors descending by index so that we can easily remove the neighbors
        incompatible_neighbors.sort_unstable_by(|(_, i_1), (_, i_2)| i_2.cmp(i_1));
        self.topo_order
            .shift_remove_bulk(incompatible_neighbors.iter().map(|(u, _)| u));
        for (neighbor, index_of_neighbor) in incompatible_neighbors {
            debug_assert!(!self.fvs.contains(&neighbor));

            self.fvs.insert(neighbor);

            // shift position of new node if neighbors to the left of position are removed
            if index_of_neighbor < position {
                position -= 1;
            }
        }

        self.topo_order.insert_at(position, node); //TODO: fix_successors is called a second time here!
        self.fvs.swap_remove(&node);
    }

    fn graph(&self) -> &'a G {
        self.graph
    }

    fn get_index(&self, value: &Node) -> Option<usize> {
        self.topo_order.iter().position(|v| v == value)
    }

    fn len(&self) -> usize {
        self.topo_order.len()
    }

    fn as_slice(&self) -> &[Node] {
        self.topo_order.as_slice()
    }

    fn fvs(&self) -> &[Node] {
        self.fvs.as_slice()
    }
}

#[cfg(test)]
mod tests_vec_topo_config {
    use super::super::macros::topo_config_base_tests;
    use super::*;

    fn factory<G: TopoGraph>(graph: &G) -> SetTopoConfig<G> {
        SetTopoConfig::new(graph)
    }

    topo_config_base_tests!(factory);
}
