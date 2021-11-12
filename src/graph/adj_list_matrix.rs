use crate::bitset::BitSet;
use std::fmt::{Debug, Formatter};
use super::*;

#[derive(Clone)]
pub struct AdjListMatrix {
    n: usize,
    out_neighbors: Vec<Vec<Node>>,
    out_matrix: Vec<BitSet>,
    in_neighbors: Vec<Vec<Node>>,
    in_matrix: Vec<BitSet>,
}

impl GraphOrder for AdjListMatrix {
    fn order(&self) -> Node { self.n as Node }
}

impl AdjecencyList for AdjListMatrix {
    fn out_neighbors(&self, u: Node) -> &[Node] {
        &self.out_neighbors[u as usize]
    }

    fn in_neighbors(&self, u: Node) -> &[Node] {
        &self.in_neighbors[u as usize]
    }
}

impl GraphManipulation for AdjListMatrix {
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(u < self.n as Node);
        assert!(v < self.n as Node);
        assert!(!self.out_matrix[u as usize][v as usize]);
        assert!(!self.in_matrix[v as usize][u as usize]);
        self.out_neighbors[u as usize].push(v);
        self.in_neighbors[v as usize].push(u);
        self.out_matrix[u as usize].set_bit(v as usize);
        self.in_matrix[v as usize].set_bit(u as usize);
    }

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(u < self.n as Node);
        assert!(v < self.n as Node);
        assert!(self.out_matrix[u as usize][v as usize]);
        assert!(self.in_matrix[v as usize][u as usize]);

        let remove = |nb : &mut Vec<Node>, v| {
            nb.swap_remove(nb.iter().enumerate().find(|(_, x)| **x == v).unwrap().0);
        };

        remove(&mut self.out_neighbors[u as usize], v);
        remove(&mut self.out_neighbors[v as usize], u);
        self.out_matrix[u as usize].unset_bit(v as usize);
        self.in_matrix[v as usize].unset_bit(u as usize);
    }
}

impl AdjecencyTest for AdjListMatrix {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_matrix[u as usize][v as usize]
    }
}

impl Debug for AdjListMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "AdjListMatrix with {} vertices", self.order())?;
        for v in self.vertices() {
            for u in self.out_neighbors(v) {
                writeln!(f, "{} -> {}", v, u,)?;
            }
        }
        Ok(())
    }
}

impl GraphNew for AdjListMatrix {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            n,
            out_neighbors: vec![vec![]; n],
            out_matrix: vec![BitSet::new(n); n],
            in_neighbors: vec![vec![]; n],
            in_matrix: vec![BitSet::new(n); n],
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn graph_edges() {
        let mut edges = vec![(1,2), (1,0), (4,3), (0,5), (2,4), (5, 4)];
        let mut graph = AdjListMatrix::new(6);
        graph.add_edges(&edges);
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }
}