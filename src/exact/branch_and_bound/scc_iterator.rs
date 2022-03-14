use super::*;
use num::{One, Zero};

pub(super) struct SCCIterator<'a, G: BBGraph> {
    transitive_closure: &'a G,
    added_to_an_scc: G::NodeMask,
    original_graph_nodes_mask: G::NodeMask,
}

impl<'a, G: BBGraph> SCCIterator<'a, G> {
    pub(super) fn new(transitive_closure: &'a G) -> Self {
        Self {
            transitive_closure,
            added_to_an_scc: G::NodeMask::zero(),
            original_graph_nodes_mask: transitive_closure.nodes_mask(),
        }
    }

    /// Returns the SCC containing `start_node`, where `start_node` is the minimal id included
    /// in the SCC.
    fn scc(&mut self, start_node: Node) -> G::NodeMask {
        // all nodes reachable from start_node with id > start_node
        let mut candidates = ((self.transitive_closure.out_neighbors(start_node))
            >> (start_node as usize + 1)) // drop all entries with id smaller/equal start node
            << (start_node as usize + 1); // restore original indices

        // mask out nodes already returned as part of an SCC
        candidates = candidates & !self.added_to_an_scc;

        let start_node_mask = G::NodeMask::one() << (start_node as usize);
        let this_scc = candidates
            .iter_ones()
            .map(|u| (u as Node, self.transitive_closure.out_neighbors(u as Node)))
            .filter(|(_, reachable)| *reachable & start_node_mask != G::NodeMask::zero())
            .map(|(u, _)| G::NodeMask::one() << (u as usize))
            .fold(start_node_mask, |a, b| a | b);

        self.added_to_an_scc |= this_scc;

        this_scc
    }
}

impl<'a, G: BBGraph> Iterator for SCCIterator<'a, G> {
    type Item = G::NodeMask;

    fn next(&mut self) -> Option<Self::Item> {
        if self.added_to_an_scc == self.original_graph_nodes_mask {
            return None;
        }

        let start_node = self.added_to_an_scc.trailing_ones() as Node;
        let next_scc = self.scc(start_node);

        Some(next_scc)
    }
}
