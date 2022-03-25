use super::macros::impl_helper_topo_new_with_fvs;
use super::topo_config::{TopoConfig, TopoGraph, TopoMove};
use crate::graph::Node;
use crate::heuristics::utils::set_vec::HashSetVec;
use std::iter::FromIterator;

/// Wraps a [`Vec<Node>`] for the topological sorting.
#[derive(Clone)]
pub struct VecTopoConfig<'a, G> {
    graph: &'a G,
    topo_order: Vec<Node>,
    fvs: HashSetVec<Node>,
}

impl<'a, G> VecTopoConfig<'a, G>
where
    G: TopoGraph,
{
    pub fn new(graph: &'a G) -> Self {
        Self {
            topo_order: Vec::with_capacity(graph.len()),
            fvs: HashSetVec::from_graph(graph),
            graph,
        }
    }

    impl_helper_topo_new_with_fvs!();
}

impl<'a, G> TopoConfig<'a, G> for VecTopoConfig<'a, G>
where
    G: TopoGraph,
{
    fn set_state<I>(&mut self, topo_order: I, fvs: I)
    where
        I: IntoIterator<Item = Node>,
    {
        self.topo_order = topo_order.into_iter().collect();
        self.fvs = HashSetVec::from_iter(fvs.into_iter());
    }

    fn perform_move(&mut self, topo_move: TopoMove) {
        debug_assert!(self.fvs().contains(&topo_move.node()));

        let (node, mut position, _, mut conflicts) = topo_move.consume(self);

        // sort neighbors descending by index so that we can easily remove the neighbors
        conflicts.sort_unstable_by(|(_, i_1), (_, i_2)| i_2.cmp(i_1));
        for (neighbor, index_of_neighbor) in conflicts.clone() {
            debug_assert!(!self.fvs.contains(&neighbor));

            self.topo_order.remove(index_of_neighbor);
            self.fvs.insert(neighbor);

            // shift position of new node if neighbors to the left of position are removed
            if index_of_neighbor < position {
                position -= 1;
            }
        }

        self.topo_order.insert(position, node);
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
        &self.topo_order
    }

    fn fvs(&self) -> &[Node] {
        self.fvs.as_slice()
    }
}

#[cfg(test)]
mod tests_vec_topo_config {
    use super::super::macros::topo_config_base_tests;
    use super::*;

    fn factory<G: TopoGraph>(graph: &G) -> VecTopoConfig<G> {
        VecTopoConfig::new(graph)
    }

    topo_config_base_tests!(factory);
}
