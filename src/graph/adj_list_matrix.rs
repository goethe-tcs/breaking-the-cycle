use super::io::DotWrite;
use super::*;
use crate::graph::adj_array::AdjArray;
use crate::graph::matrix::AdjMatrix;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops,
/// but no multi-edges. Supports constant time edge-existence queries
/// Stores no in_edges at each vertex, for this refer to AdjListMatrixIn
#[derive(Clone)]
pub struct AdjListMatrix {
    adj_matrix: AdjMatrix,
    adj_array: AdjArray,
}

#[derive(Clone)]
pub struct AdjListMatrixIn {
    adj_out: AdjListMatrix,
    adj_in: AdjListMatrix,
}

graph_macros::impl_helper_graph_debug!(AdjListMatrix);
graph_macros::impl_helper_graph_from_edges!(AdjListMatrix);
graph_macros::impl_helper_graph_debug!(AdjListMatrixIn);
graph_macros::impl_helper_graph_from_edges!(AdjListMatrixIn);

graph_macros::impl_helper_adjacency_list!(AdjListMatrix, adj_array);
graph_macros::impl_helper_graph_order!(AdjListMatrix, adj_array);
graph_macros::impl_helper_adjacency_test!(AdjListMatrix, adj_matrix);

graph_macros::impl_helper_adjacency_list!(AdjListMatrixIn, adj_out);
graph_macros::impl_helper_graph_order!(AdjListMatrixIn, adj_out);
graph_macros::impl_helper_adjacency_test!(AdjListMatrixIn, adj_out);

impl GraphEdgeEditing for AdjListMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj_array.add_edge(u, v);
        self.adj_matrix.add_edge(u, v);
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj_array.remove_edge(u, v);
        self.adj_matrix.remove_edge(u, v);
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
}

impl GraphEdgeEditing for AdjListMatrixIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj_out.add_edge(u, v);
        self.adj_in.add_edge(v, u);
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj_out.remove_edge(u, v);
        self.adj_in.remove_edge(v, u);
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
pub mod tests {
    use super::*;

    #[test]
    fn graph_edges() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = AdjListMatrix::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn test_remove_edges() {
        let org_graph = AdjListMatrix::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

        // no changes
        {
            let mut graph = org_graph.clone();

            graph.remove_edges_into_node(0);
            assert_eq!(graph.edges(), org_graph.edges());

            graph.remove_edges_out_of_node(4);
            assert_eq!(graph.edges(), org_graph.edges());
        }

        // remove out
        {
            let mut graph = org_graph.clone();

            graph.remove_edges_out_of_node(3);
            assert_eq!(
                graph.number_of_edges(),
                org_graph.number_of_edges() - org_graph.out_degree(3) as usize
            );
            for (u, _v) in graph.edges() {
                assert_ne!(u, 3);
            }
        }

        // remove in
        {
            let mut graph = org_graph.clone();

            let in_degree = graph.vertices().filter(|v| graph.has_edge(*v, 3)).count();
            graph.remove_edges_into_node(3);
            assert_eq!(
                graph.number_of_edges(),
                org_graph.number_of_edges() - in_degree
            );
            for (_u, v) in graph.edges() {
                assert_ne!(v, 3);
            }
        }
    }

    #[test]
    fn test_debug_format() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }
}
