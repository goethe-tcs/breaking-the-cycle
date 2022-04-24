use super::graph_macros::*;
use super::io::DotWrite;
use super::*;
use crate::graph::adj_array::AdjArray;
use crate::graph::matrix::AdjMatrix;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops,
/// but no multi-edges. Supports constant time edge-existence queries
/// Stores no in_edges at each vertex, for this refer to AdjListMatrixIn
#[derive(Clone, Default)]
pub struct AdjListMatrix {
    adj_matrix: AdjMatrix,
    adj_array: AdjArray,
}

#[derive(Clone, Default)]
pub struct AdjListMatrixIn {
    adj_out: AdjListMatrix,
    adj_in: AdjListMatrix,
}

graph_macros::impl_helper_graph_debug!(AdjListMatrix);
graph_macros::impl_helper_graph_from_edges!(AdjListMatrix);
graph_macros::impl_helper_graph_debug!(AdjListMatrixIn);
graph_macros::impl_helper_graph_from_edges!(AdjListMatrixIn);
graph_macros::impl_helper_undir_adjacency_test!(AdjListMatrix);
graph_macros::impl_helper_graph_from_slice!(AdjListMatrix);

graph_macros::impl_helper_adjacency_list!(AdjListMatrix, adj_array);
graph_macros::impl_helper_adjacency_test!(AdjListMatrix, adj_matrix);
graph_macros::impl_helper_graph_order!(AdjListMatrix, adj_array);
graph_macros::impl_helper_adjacency_list!(AdjListMatrixIn, adj_out);
graph_macros::impl_helper_adjacency_test_linear_search_bi_directed!(AdjListMatrixIn);
graph_macros::impl_helper_graph_order!(AdjListMatrixIn, adj_out);
graph_macros::impl_helper_undir_adjacency_test!(AdjListMatrixIn);
graph_macros::impl_helper_adjacency_list_undir!(AdjListMatrixIn);
graph_macros::impl_helper_graph_from_slice!(AdjListMatrixIn);

impl GraphEdgeEditing for AdjListMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj_array.add_edge(u, v);
        self.adj_matrix.add_edge(u, v);
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj_array.remove_edge(u, v);
        self.adj_matrix.remove_edge(u, v);
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        self.adj_array.try_remove_edge(u, v) && self.adj_matrix.try_remove_edge(u, v)
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        self.adj_array.remove_edges_into_node(u);
        self.adj_matrix.remove_edges_into_node(u);
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.adj_array.remove_edges_out_of_node(u);
        self.adj_matrix.remove_edges_out_of_node(u);
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        self.adj_array.contract_node(u);
        self.adj_matrix.contract_node(u)
    }
}

impl GraphEdgeEditing for AdjListMatrixIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj_out.add_edge(u, v);
        self.adj_in.add_edge(v, u);
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj_out.remove_edge(u, v);
        self.adj_in.remove_edge(v, u);
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        self.adj_out.try_remove_edge(u, v) && self.adj_in.try_remove_edge(v, u)
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        // contains all edges into u
        let edges: Vec<_> = self.adj_in.adj_array.out_neighbors(u).collect();
        for v in edges {
            self.remove_edge(v, u);
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        // contains all edges into u
        let edges: Vec<_> = self.adj_out.adj_array.out_neighbors(u).collect();
        for v in edges {
            self.remove_edge(u, v);
        }
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        self.adj_in.contract_node(u);
        self.adj_out.contract_node(u)
    }
}

impl GraphNew for AdjListMatrix {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            adj_matrix: AdjMatrix::new(n),
            adj_array: AdjArray::new(n),
        }
    }
}

impl GraphNew for AdjListMatrixIn {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            adj_in: AdjListMatrix::new(n),
            adj_out: AdjListMatrix::new(n),
        }
    }
}

impl AdjacencyListIn for AdjListMatrixIn {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_> {
        self.adj_in.out_neighbors(u)
    }

    fn in_degree(&self, u: Node) -> Node {
        self.adj_in.out_degree(u)
    }
}

#[cfg(test)]
pub mod tests_adj_list_matrix {
    use super::graph_macros::base_tests;
    use super::*;
    base_tests!(AdjListMatrix);
}

#[cfg(test)]
pub mod tests_adj_list_matrix_in {
    use super::graph_macros::base_tests_in;
    use super::*;
    base_tests_in!(AdjListMatrixIn);
}
