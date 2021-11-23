use super::io::DotWrite;
use super::*;
use crate::bitset::BitSet;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops,
/// but no multi-edges. Supports constant time edge-existence queries
#[derive(Clone)]
pub struct AdjMatrix {
    n: usize,
    m: usize,
    out_matrix: Vec<BitSet>,
}

/// Same as AdjMatrix, but stores a matrix of all in-edges as well
#[derive(Clone)]
pub struct AdjMatrixIn {
    adj: AdjMatrix,
    in_matrix: Vec<BitSet>,
}

graph_macros::impl_helper_graph_debug!(AdjMatrix);
graph_macros::impl_helper_graph_from_edges!(AdjMatrix);

graph_macros::impl_helper_graph_debug!(AdjMatrixIn);
graph_macros::impl_helper_graph_from_edges!(AdjMatrixIn);
graph_macros::impl_helper_adjacency_list!(AdjMatrixIn, adj);
graph_macros::impl_helper_graph_order!(AdjMatrixIn, adj);
graph_macros::impl_helper_adjacency_test!(AdjMatrixIn, adj);

impl GraphOrder for AdjMatrix {
    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }
    fn number_of_edges(&self) -> usize {
        self.m
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
        self.in_matrix[u as usize].len() as Node
    }
}

impl GraphEdgeEditing for AdjMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(!self.out_matrix[u as usize][v as usize]);
        self.out_matrix[u as usize].set_bit(v as usize);
        self.m += 1;
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(u < self.n as Node);
        assert!(v < self.n as Node);
        assert!(self.out_matrix[u as usize][v as usize]);
        assert!(self.m > 0);
        self.out_matrix[u as usize].unset_bit(v as usize);
        self.m -= 1;
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.vertices() {
            if self.out_matrix[v as usize][u as usize] {
                self.out_matrix[v as usize].unset_bit(u as usize);
                self.m -= 1;
            }
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.m -= self.out_matrix[u as usize].cardinality();
        self.out_matrix[u as usize] = BitSet::new(self.n);
    }
}

impl GraphEdgeEditing for AdjMatrixIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj.add_edge(u, v);
        self.in_matrix[v as usize].set_bit(u as usize);
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj.remove_edge(u, v);
        self.in_matrix[v as usize].unset_bit(u as usize);
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.in_matrix[u as usize].iter() {
            self.adj.remove_edge(v as Node, u);
        }
        self.in_matrix[u as usize] = BitSet::new(self.adj.n);
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.adj.m -= self.adj.out_matrix[u as usize].cardinality();
        let mut out = BitSet::new(self.adj.n);
        std::mem::swap(&mut out, &mut self.adj.out_matrix[u as usize]);
        for v in out.iter() {
            self.in_matrix[v].unset_bit(u as usize);
        }
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
pub mod tests {
    use super::*;

    #[test]
    fn graph_edges() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = AdjMatrix::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn test_remove_edges() {
        let org_graph = AdjMatrix::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

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

            let in_degree = 3;
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
        let mut g = AdjMatrix::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        println!("{}", &str);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }
}
