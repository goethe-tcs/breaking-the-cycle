use super::graph_macros::*;
use super::io::DotWrite;
use super::*;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops and multi-edges
#[derive(Clone, Default)]
pub struct AdjArray {
    n: usize,
    m: usize,
    pub(super) out_neighbors: Vec<Vec<Node>>,
}

/// Same as AdjArray, but stores all in-edges for each vertex in addition to the outedges
#[derive(Clone, Default)]
pub struct AdjArrayIn {
    in_neighbors: Vec<Vec<Node>>,
    adj: AdjArray,
}

graph_macros::impl_helper_graph_debug!(AdjArray);
graph_macros::impl_helper_graph_from_edges!(AdjArray);
graph_macros::impl_helper_undir_adjacency_test!(AdjArray);

graph_macros::impl_helper_graph_order!(AdjArrayIn, adj);
graph_macros::impl_helper_graph_debug!(AdjArrayIn);
graph_macros::impl_helper_graph_from_edges!(AdjArrayIn);
graph_macros::impl_helper_adjacency_list!(AdjArrayIn, adj);
graph_macros::impl_helper_adjacency_test_linear_search_bi_directed!(AdjArrayIn);
graph_macros::impl_helper_undir_adjacency_test!(AdjArrayIn);
graph_macros::impl_helper_adjacency_list_undir!(AdjArrayIn);

impl GraphOrder for AdjArray {
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

impl AdjacencyList for AdjArray {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.out_neighbors[u as usize].iter().copied()
    }

    fn out_degree(&self, u: Node) -> Node {
        self.out_neighbors[u as usize].len() as Node
    }
}

impl AdjacencyListIn for AdjArrayIn {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_> {
        self.in_neighbors[u as usize].iter().copied()
    }

    fn in_degree(&self, u: Node) -> Node {
        self.in_neighbors[u as usize].len() as u32
    }
}

fn remove_helper(nb: &mut Vec<Node>, v: Node) {
    nb.swap_remove(nb.iter().enumerate().find(|(_, x)| **x == v).unwrap().0);
}

fn try_remove_helper(nb: &mut Vec<Node>, v: Node) -> Result<(), ()> {
    return if let Some((i, _)) = nb.iter().enumerate().find(|(_, x)| **x == v) {
        nb.swap_remove(i);
        Ok(())
    } else {
        Err(())
    };
}

impl GraphEdgeEditing for AdjArray {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.out_neighbors[u as usize].push(v);
        self.m += 1;
    }
    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        remove_helper(&mut self.out_neighbors[u as usize], v);
        self.m -= 1;
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if try_remove_helper(&mut self.out_neighbors[u as usize], v).is_ok() {
            self.m -= 1;
            true
        } else {
            false
        }
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for v in self.vertices_range() {
            if try_remove_helper(&mut self.out_neighbors[v as usize], u).is_ok() {
                self.m -= 1;
            };
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        self.m -= self.out_neighbors[u as usize].len();
        self.out_neighbors[u as usize].clear();
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        let mut loops = Vec::new();
        self.try_remove_edge(u, u);
        let out_neighbors = std::mem::take(&mut self.out_neighbors[u as usize]);

        for v in self.vertices_range() {
            if try_remove_helper(&mut self.out_neighbors[v as usize], u).is_ok() {
                self.m -= 1;

                for &w in &out_neighbors {
                    self.try_add_edge(v, w);

                    if v == w {
                        loops.push(v);
                    }
                }
            };
        }

        self.m -= out_neighbors.len();

        loops
    }
}

impl GraphEdgeEditing for AdjArrayIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj.add_edge(u, v);
        self.in_neighbors[v as usize].push(u);
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj.remove_edge(u, v);
        remove_helper(&mut self.in_neighbors[v as usize], u);
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        self.adj.try_remove_edge(u, v)
            && try_remove_helper(&mut self.in_neighbors[v as usize], u).is_ok()
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        for &v in &self.in_neighbors[u as usize] {
            remove_helper(&mut self.adj.out_neighbors[v as usize], u);
        }
        self.adj.m -= self.in_neighbors[u as usize].len();
        self.in_neighbors[u as usize].clear();
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        for &v in &self.adj.out_neighbors[u as usize] {
            remove_helper(&mut self.in_neighbors[v as usize], u);
        }
        self.adj.m -= self.adj.out_neighbors[u as usize].len();
        self.adj.out_neighbors[u as usize].clear();
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        let mut loops = Vec::new();

        self.try_remove_edge(u, u);
        let in_neighbors = std::mem::take(&mut self.in_neighbors[u as usize]);
        let out_neighbors = std::mem::take(&mut self.adj.out_neighbors[u as usize]);

        for &w in &out_neighbors {
            remove_helper(&mut self.in_neighbors[w as usize], u);
        }

        for &v in &in_neighbors {
            remove_helper(&mut self.adj.out_neighbors[v as usize], u);
            for &w in &out_neighbors {
                self.try_add_edge(v, w);
                if v == w {
                    loops.push(v);
                }
            }
        }

        self.adj.m -= in_neighbors.len() + out_neighbors.len();
        loops
    }
}

impl AdjacencyTest for AdjArray {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.out_neighbors[u as usize].iter().any(|w| *w == v)
    }
}

impl GraphNew for AdjArray {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            n,
            m: 0,
            out_neighbors: vec![vec![]; n],
        }
    }
}

impl GraphNew for AdjArrayIn {
    /// Creates a new AdjListMatrix with *V={0,1,...,n-1}* and without any edges.
    fn new(n: usize) -> Self {
        Self {
            adj: AdjArray::new(n),
            in_neighbors: vec![vec![]; n],
        }
    }
}

#[cfg(test)]
pub mod tests_adj_array_in {
    use super::graph_macros::base_tests_in;
    use super::*;
    base_tests_in!(AdjArrayIn);
}

#[cfg(test)]
pub mod tests_adj_array {
    use super::graph_macros::base_tests;
    use super::*;
    base_tests!(AdjArray);
}
