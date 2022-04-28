use super::graph_macros::*;
use super::io::DotWrite;
use super::*;
use crate::bitset::BitSet;
use itertools::Itertools;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops,
/// but no multi-edges. Supports constant time edge-existence queries
#[derive(Clone, Default)]
pub struct AdjMatrix {
    n: usize,
    m: usize,
    out_matrix: Vec<BitSet>,
}

/// Same as AdjMatrix, but stores a matrix of all in-edges as well
#[derive(Clone, Default)]
pub struct AdjMatrixIn {
    adj: AdjMatrix,
    in_matrix: Vec<BitSet>,
}

graph_macros::impl_helper_graph_debug!(AdjMatrix);
graph_macros::impl_helper_graph_from_edges!(AdjMatrix);
graph_macros::impl_helper_undir_adjacency_test!(AdjMatrix);
graph_macros::impl_helper_graph_from_slice!(AdjMatrix);

graph_macros::impl_helper_graph_debug!(AdjMatrixIn);
graph_macros::impl_helper_graph_from_edges!(AdjMatrixIn);
graph_macros::impl_helper_adjacency_list!(AdjMatrixIn, adj);
graph_macros::impl_helper_adjacency_test!(AdjMatrixIn, adj);
graph_macros::impl_helper_graph_order!(AdjMatrixIn, adj);
graph_macros::impl_helper_undir_adjacency_test!(AdjMatrixIn);
graph_macros::impl_helper_adjacency_list_undir!(AdjMatrixIn);
graph_macros::impl_helper_graph_from_slice!(AdjMatrixIn);

impl GraphOrder for AdjMatrix {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }
    fn number_of_edges(&self) -> usize {
        self.m
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }
}

impl AdjacencyList for AdjMatrix {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.out_matrix[u as usize].iter().map(|v| v as Node)
    }

    fn out_degree(&self, u: Node) -> Node {
        self.out_matrix[u as usize].cardinality() as Node
    }
}

impl AdjacencyListIn for AdjMatrixIn {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: u32) -> Self::IterIn<'_> {
        self.in_matrix[u as usize].iter().map(|v| v as Node)
    }

    fn in_degree(&self, u: u32) -> u32 {
        self.in_matrix[u as usize].cardinality() as Node
    }
}

impl GraphEdgeEditing for AdjMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        debug_assert!(!self.out_matrix[u as usize][v as usize]);
        self.out_matrix[u as usize].set_bit(v as usize);
        self.m += 1;
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.out_matrix[u as usize][v as usize]);
        self.out_matrix[u as usize].unset_bit(v as usize);
        self.m -= 1;
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if self.out_matrix[u as usize].unset_bit(v as usize) {
            self.m -= 1;
            true
        } else {
            false
        }
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.vertices_range() {
            self.m -= self.out_matrix[v as usize].unset_bit(u as usize) as usize;
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.m -= self.out_matrix[u as usize].cardinality();
        self.out_matrix[u as usize].unset_all();
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        self.try_remove_edge(u, u);

        if self.out_degree(u) == 0 {
            self.remove_edges_into_node(u);
            return vec![];
        }

        self.m -= self.out_matrix[u as usize].cardinality();
        let old_out_neighbors = {
            let mut empty_row = BitSet::new(self.n);
            std::mem::swap(&mut self.out_matrix[u as usize], &mut empty_row);
            empty_row
        };

        let mut loops = Vec::new();
        for v in self.vertices_range() {
            if self.out_matrix[v as usize].unset_bit(u as usize) {
                self.m -= 1;

                for w in old_out_neighbors.iter() {
                    self.try_add_edge(v, w as Node);
                    if v == w as Node {
                        loops.push(v);
                    }
                }
            }
        }

        debug_assert_eq!(self.out_matrix[u as usize].cardinality(), 0);
        loops
    }
}

impl GraphEdgeEditing for AdjMatrixIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj.add_edge(u, v);
        self.in_matrix[v as usize].set_bit(u as usize);
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj.remove_edge(u, v);
        self.in_matrix[v as usize].unset_bit(u as usize);
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        self.adj.try_remove_edge(u, v) && self.in_matrix[v as usize].unset_bit(u as usize)
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.in_matrix[u as usize].iter() {
            self.adj.remove_edge(v as Node, u);
        }
        self.in_matrix[u as usize].unset_all();
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.adj.m -= self.adj.out_matrix[u as usize].cardinality();
        let out = &mut self.adj.out_matrix[u as usize];
        for v in out.iter() {
            self.in_matrix[v].unset_bit(u as usize);
        }
        out.unset_all();
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        let in_neighbors = self.in_matrix[u as usize]
            .iter()
            .map(|i| i as Node)
            .collect_vec();
        let out_neighbors = self.adj.out_matrix[u as usize]
            .iter()
            .map(|i| i as Node)
            .collect_vec();

        let mut loops = Vec::new();
        for v in in_neighbors {
            for &w in &out_neighbors {
                self.try_add_edge(v, w);
                if v == w {
                    loops.push(v);
                }
            }
        }

        self.remove_edges_at_node(u);
        loops
    }
}

impl AdjacencyTest for AdjMatrix {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_matrix[u as usize][v as usize]
    }
}

impl GraphNew for AdjMatrix {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            n,
            m: 0,
            out_matrix: vec![BitSet::new(n); n],
        }
    }
}

impl GraphNew for AdjMatrixIn {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            adj: AdjMatrix::new(n),
            in_matrix: vec![BitSet::new(n); n],
        }
    }
}

#[cfg(test)]
pub mod tests_adj_matrix {
    use super::graph_macros::base_tests;
    use super::*;
    base_tests!(AdjMatrix);
}

#[cfg(test)]
pub mod tests_adj_matrix_in {
    use super::graph_macros::base_tests_in;
    use super::*;
    base_tests_in!(AdjMatrixIn);
}
