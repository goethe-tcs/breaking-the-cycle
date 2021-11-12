use crate::bitset::BitSet;
use std::cmp::min;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::ops::Range;

#[derive(Clone)]
pub struct Graph {
    n: usize,
    out_neighbors: Vec<Vec<u32>>,
    out_matrix: Vec<BitSet>,
    in_neighbors: Vec<Vec<u32>>,
    in_matrix: Vec<BitSet>,
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph with {} vertices", self.order())?;
        for v in self.vertices() {
            for u in self.out_neighbors(v) {
                writeln!(f, "{} -> {}", v, u,)?;
            }
        }
        Ok(())
    }
}

impl Graph {
    /// Creates a new graph with *V={0,1,...,n-1}* and without any edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            out_neighbors: vec![vec![]; n],
            out_matrix: vec![BitSet::new(n); n],
            in_neighbors: vec![vec![]; n],
            in_matrix: vec![BitSet::new(n); n],
        }
    }

    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is already contained or u, v >= n **
    pub fn add_edge(&mut self, u: u32, v: u32) {
        assert!(u < self.n as u32);
        assert!(v < self.n as u32);
        assert!(!self.out_matrix[u as usize][v as usize]);
        assert!(!self.in_matrix[v as usize][u as usize]);
        self.out_neighbors[u as usize].push(v);
        self.in_neighbors[v as usize].push(u);
        self.out_matrix[u as usize].set_bit(v as usize);
        self.in_matrix[v as usize].set_bit(u as usize);
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    pub fn remove_edge(&mut self, u: u32, v: u32) {
        assert!(u < self.n as u32);
        assert!(v < self.n as u32);
        assert!(self.out_matrix[u as usize][v as usize]);
        assert!(self.in_matrix[v as usize][u as usize]);
        Self::remove_edge_helper(&mut self.out_neighbors[u as usize], v);
        Self::remove_edge_helper(&mut self.out_neighbors[v as usize], u);
        self.out_matrix[u as usize].unset_bit(v as usize);
        self.out_matrix[v as usize].unset_bit(u as usize);
    }

    fn remove_edge_helper(nb: &mut Vec<u32>, v: u32) {
        nb.swap_remove(nb.iter().enumerate().find(|(_, x)| **x == v).unwrap().0);
    }

    /// Returns a slice over the outgoing neighbors of a given vertex.
    /// ** Panics if the v >= n **
    pub fn out_neighbors(&self, u: u32) -> &[u32] {
        assert!(u < self.n as u32);
        &self.out_neighbors[u as usize]
    }

    /// Returns a slice over the ingoing neighbors of a given vertex.
    /// ** Panics if the v >= n **
    pub fn in_neighbors(&self, u: u32) -> &[u32] {
        assert!(u < self.n as u32);
        &self.in_neighbors[u as usize]
    }

    /// Returns an iterator over V.
    pub fn vertices(&self) -> Range<u32> {
        0..self.order()
    }

    /// Returns the order of the graph
    pub fn order(&self) -> u32 {
        self.n as u32
    }

    /// Returns *true* exactly if the graph contains the directed edge (u, v)
    pub fn has_edge(&self, u: u32, v: u32) -> bool {
        self.out_matrix[u as usize][v as usize]
    }

    /// Adds all edges in the collection
    pub fn add_edges<'a, T: IntoIterator<Item = &'a (u32, u32)>>(&'a mut self, edges: T) {
        for e in edges {
            self.add_edge(e.0, e.1);
        }
    }

    /// Returns the number of ingoing edges to *u*
    pub fn in_degree(&self, u: u32) -> u32 {
        self.in_neighbors[u as usize].len() as u32
    }

    /// Returns the number of outgoing edges from *u*
    pub fn out_degree(&self, u: u32) -> u32 {
        self.out_neighbors[u as usize].len() as u32
    }

    /// Returns the total number of edges incident to *u*
    pub fn total_degree(&self, u: u32) -> u32 {
        self.in_degree(u) + self.out_degree(u)
    }

    /// Returns the strongly connected components of the graph as a Vec<Vec<u32>>
    pub fn strongly_connected_components(&self) -> Vec<Vec<u32>> {
        let sc = StronglyConnected::new(self);
        sc.find()
    }

    /// Returns the connected components of the graph as BitSets
    pub fn connected_components(&self) -> Vec<BitSet> {
        let mut unvisited: HashSet<_> = self.vertices().collect();
        let mut components: Vec<BitSet> = vec![];
        while let Some(v) = unvisited.iter().copied().next() {
            let mut stack = vec![v];
            unvisited.remove(&v);
            let mut component = BitSet::new(self.order() as usize);
            while let Some(u) = stack.pop() {
                for w in self.in_neighbors[u as usize]
                    .iter()
                    .chain(self.out_neighbors[u as usize].iter())
                {
                    if unvisited.contains(w) {
                        unvisited.remove(w);
                        stack.push(*w);
                        component.set_bit(*w as usize);
                    }
                }
            }
            components.push(component);
        }
        components
    }
}

struct StronglyConnected<'a> {
    graph: &'a Graph,
    idx: u32,
    stack: Vec<u32>,
    indices: Vec<Option<u32>>,
    low_links: Vec<u32>,
    on_stack: BitSet,
    components: Vec<Vec<u32>>,
}

impl<'a> StronglyConnected<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            idx: 0,
            stack: vec![],
            indices: vec![None; graph.n],
            low_links: vec![0; graph.n],
            on_stack: BitSet::new(graph.n),
            components: vec![],
        }
    }

    pub fn find(mut self) -> Vec<Vec<u32>> {
        for v in self.graph.vertices() {
            if self.indices[v as usize].is_none() {
                self.sc(v);
            }
        }
        self.components
    }

    fn sc(&mut self, v: u32) {
        self.indices[v as usize] = Some(self.idx);
        self.low_links[v as usize] = self.idx;
        self.idx += 1;
        self.stack.push(v);
        self.on_stack.set_bit(v as usize);

        for w in self.graph.out_neighbors(v) {
            if self.indices[*w as usize].is_none() {
                self.sc(*w);
                self.low_links[v as usize] =
                    min(self.low_links[v as usize], self.low_links[*w as usize]);
            } else if self.on_stack[*w as usize] {
                self.low_links[v as usize] = min(
                    self.low_links[v as usize],
                    self.indices[*w as usize].unwrap(),
                );
            }
        }

        if self.low_links[v as usize] == self.indices[v as usize].unwrap() {
            // found SC
            let mut component = Vec::with_capacity(self.graph.order() as usize);
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack.unset_bit(w as usize);
                component.push(w);
                if w == v {
                    break;
                }
            }
            self.components.push(component);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::Graph;

    #[test]
    pub fn scc() {
        let mut graph = Graph::new(8);
        graph.add_edges(&[
            (0, 1),
            (1, 5),
            (1, 4),
            (1, 3),
            (2, 6),
            (2, 3),
            (3, 2),
            (3, 7),
            (4, 0),
            (4, 5),
            (5, 6),
            (6, 5),
            (7, 3),
            (7, 6),
        ]);

        let mut sccs = graph.strongly_connected_components();
        assert_eq!(sccs.len(), 3);
        assert!(!sccs[0].is_empty());
        assert!(!sccs[1].is_empty());
        assert!(!sccs[2].is_empty());

        for mut scc in &mut sccs {
            scc.sort();
        }
        sccs.sort_by(|a, b| a[0].cmp(&b[0]));
        assert_eq!(sccs[0], [0, 1, 4]);
        assert_eq!(sccs[1], [2, 3, 7]);
        assert_eq!(sccs[2], [5, 6]);
    }
}
