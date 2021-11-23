use super::io::DotWrite;
use super::*;
use crate::bitset::BitSet;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops,
/// but no multi-edges. Supports constant time edge-existence queries
#[derive(Clone)]
pub struct AdjListMatrix {
    n: usize,
    m: usize,
    out_neighbors: Vec<Vec<Node>>,
    out_matrix: Vec<BitSet>,
    in_neighbors: Vec<Vec<Node>>,
    in_matrix: Vec<BitSet>,
}

impl AdjacencyListIn for AdjListMatrix {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_> {
        self.in_neighbors[u as usize].iter().copied()
    }

    fn in_degree(&self, u: Node) -> Node {
        self.in_neighbors[u as usize].len() as Node
    }

    fn total_degree(&self, u: Node) -> Node {
        self.in_degree(u) + self.out_degree(u)
    }
}

impl GraphOrder for AdjListMatrix {
    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }
    fn number_of_edges(&self) -> usize {
        self.m
    }
}

impl AdjacencyList for AdjListMatrix {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.out_neighbors[u as usize].iter().copied()
    }

    fn out_degree(&self, u: Node) -> Node {
        self.out_neighbors[u as usize].len() as Node
    }
}

fn remove_helper(nb: &mut Vec<Node>, v: Node) {
    nb.swap_remove(nb.iter().enumerate().find(|(_, x)| **x == v).unwrap().0);
}

impl GraphEdgeEditing for AdjListMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(u < self.n as Node);
        assert!(v < self.n as Node);
        assert!(!self.out_matrix[u as usize][v as usize]);
        assert!(!self.in_matrix[v as usize][u as usize]);
        self.out_neighbors[u as usize].push(v);
        self.in_neighbors[v as usize].push(u);
        self.out_matrix[u as usize].set_bit(v as usize);
        self.in_matrix[v as usize].set_bit(u as usize);
        self.m += 1;
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(u < self.n as Node);
        assert!(v < self.n as Node);
        assert!(self.out_matrix[u as usize][v as usize]);
        assert!(self.in_matrix[v as usize][u as usize]);
        assert!(self.m > 0);

        remove_helper(&mut self.out_neighbors[u as usize], v);
        remove_helper(&mut self.out_neighbors[v as usize], u);
        self.out_matrix[u as usize].unset_bit(v as usize);
        self.in_matrix[v as usize].unset_bit(u as usize);
        self.m -= 1;
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for &v in &self.in_neighbors[u as usize] {
            remove_helper(&mut self.out_neighbors[v as usize], u);
            self.out_matrix[v as usize].unset_bit(u as usize);
            self.in_matrix[u as usize].unset_bit(v as usize);
        }
        self.m -= self.in_neighbors[u as usize].len();
        self.in_neighbors[u as usize].clear();
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        for &v in &self.out_neighbors[u as usize] {
            remove_helper(&mut self.in_neighbors[v as usize], u);
            self.out_matrix[u as usize].unset_bit(v as usize);
            self.in_matrix[v as usize].unset_bit(u as usize);
        }
        self.m -= self.out_neighbors[u as usize].len();
        self.out_neighbors[u as usize].clear();
    }
}

impl AdjacencyTest for AdjListMatrix {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_matrix[u as usize][v as usize]
    }
}

impl GraphNew for AdjListMatrix {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            n,
            m: 0,
            out_neighbors: vec![vec![]; n],
            out_matrix: vec![BitSet::new(n); n],
            in_neighbors: vec![vec![]; n],
            in_matrix: vec![BitSet::new(n); n],
        }
    }
}

impl<'a, T: IntoIterator<Item = &'a Edge> + Clone> From<T> for AdjListMatrix {
    fn from(edges: T) -> Self {
        let n = edges
            .clone()
            .into_iter()
            .map(|e| e.0.max(e.1) + 1)
            .max()
            .unwrap_or(0);
        let mut graph = AdjListMatrix::new(n as usize);
        for e in edges {
            graph.add_edge(e.0, e.1);
        }
        graph
    }
}

impl Debug for AdjListMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = Vec::new();
        if self.try_write_dot(&mut buf).is_ok() {
            f.write_str(str::from_utf8(&buf).unwrap())?;
        }

        Ok(())
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

            graph.remove_edges_into_node(3);
            assert_eq!(
                graph.number_of_edges(),
                org_graph.number_of_edges() - org_graph.in_degree(3) as usize
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
