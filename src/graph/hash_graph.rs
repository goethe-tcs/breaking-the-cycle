use super::graph_macros::*;
use super::io::DotWrite;
use super::*;
use fxhash::{FxHashMap, FxHashSet};
use itertools::Itertools;
use std::fmt::Debug;
use std::{fmt, str};

#[derive(Clone, Default)]
pub struct HashGraph {
    adj: FxHashMap<Node, FxHashSet<Node>>,
    m: usize,
}

#[derive(Clone, Default)]
pub struct HashGraphIn {
    adj_out: HashGraph,
    adj_in: FxHashMap<Node, FxHashSet<Node>>,
}

graph_macros::impl_helper_graph_debug!(HashGraph);
graph_macros::impl_helper_graph_from_edges!(HashGraph);
graph_macros::impl_helper_undir_adjacency_test!(HashGraph);
graph_macros::impl_helper_graph_from_slice!(HashGraph);

graph_macros::impl_helper_graph_debug!(HashGraphIn);
graph_macros::impl_helper_graph_from_edges!(HashGraphIn);
graph_macros::impl_helper_adjacency_list!(HashGraphIn, adj_out);
graph_macros::impl_helper_adjacency_test!(HashGraphIn, adj_out);
graph_macros::impl_helper_graph_order!(HashGraphIn, adj_out);
graph_macros::impl_helper_undir_adjacency_test!(HashGraphIn);
graph_macros::impl_helper_adjacency_list_undir!(HashGraphIn);
graph_macros::impl_helper_graph_from_slice!(HashGraphIn);

impl GraphVertexEditing for HashGraph {
    fn remove_vertex(&mut self, u: Node) {
        let nb_out = self.adj.remove(&u).unwrap();
        let mut nb_in: Vec<_> = Default::default();
        for v in self.vertices() {
            if self.adj.get(&v).unwrap().contains(&u) {
                nb_in.push(v);
            }
        }
        self.m -= nb_out.len() + nb_in.len();
        for v in nb_in {
            self.adj.get_mut(&v).unwrap().remove(&u);
        }
    }

    fn has_vertex(&self, u: Node) -> bool {
        self.adj.contains_key(&u)
    }
}

impl GraphVertexEditing for HashGraphIn {
    fn remove_vertex(&mut self, u: Node) {
        let nb_in = self.adj_in.remove(&u).unwrap();
        let nb_out = self.adj_out.adj.remove(&u).unwrap();
        self.adj_out.m -= nb_in.len() + nb_out.len();

        // remove all edges (v, u)
        for v in nb_in {
            self.adj_out.adj.get_mut(&v).unwrap().remove(&u);
        }

        // remove all edges (u, v) from the in-neighbors of v
        for v in nb_out {
            self.adj_in.get_mut(&v).unwrap().remove(&u);
        }
    }

    fn has_vertex(&self, u: Node) -> bool {
        self.adj_in.contains_key(&u)
    }
}

impl GraphOrder for HashGraph {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> Node {
        self.adj.len() as Node
    }
    fn number_of_edges(&self) -> usize {
        self.m
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.adj.keys().copied()
    }
}

impl AdjacencyList for HashGraph {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.adj.get(&u).unwrap().iter().copied()
    }

    fn out_degree(&self, u: Node) -> Node {
        self.adj.get(&u).unwrap().len() as Node
    }
}

impl AdjacencyTest for HashGraph {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        self.adj.contains_key(&u) && self.adj.get(&u).unwrap().contains(&v)
    }
}

impl AdjacencyListIn for HashGraphIn {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_> {
        self.adj_in.get(&u).unwrap().iter().copied()
    }

    fn in_degree(&self, u: Node) -> Node {
        self.adj_in.get(&u).unwrap().len() as Node
    }
}

impl GraphEdgeEditing for HashGraph {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj
            .entry(u)
            .or_insert_with(FxHashSet::default)
            .insert(v);
        self.adj.entry(v).or_insert_with(FxHashSet::default);
        self.m += 1;
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        if let Some(nb) = self.adj.get_mut(&u) {
            if nb.remove(&v) {
                self.m -= 1;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        let mut edges: Vec<_> = Default::default();
        for v in self.vertices() {
            if self.adj.get(&v).unwrap().contains(&u) {
                edges.push(v);
            }
        }
        self.m -= edges.len();
        for v in edges {
            self.adj.get_mut(&v).unwrap().remove(&u);
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        let mut edges = FxHashSet::default();
        std::mem::swap(&mut edges, self.adj.get_mut(&u).unwrap());
        self.m -= edges.len();
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        self.try_remove_edge(u, u);

        let in_neighbors = self
            .vertices()
            .filter(|&v| self.adj.get(&v).unwrap().contains(&u))
            .collect_vec();

        let out_neighbors = std::mem::take(self.adj.get_mut(&u).unwrap());
        self.m -= in_neighbors.len() + out_neighbors.len();

        let mut loops = Vec::new();
        for v in in_neighbors {
            self.adj.get_mut(&v).unwrap().remove(&u);
            for &w in out_neighbors.iter() {
                self.try_add_edge(v, w);
                if v == w {
                    loops.push(v);
                }
            }
        }

        loops
    }
}

impl GraphEdgeEditing for HashGraphIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj_out.add_edge(u, v);
        self.adj_in
            .entry(v)
            .or_insert_with(FxHashSet::default)
            .insert(u);
        self.adj_in.entry(u).or_insert_with(FxHashSet::default);
    }

    impl_helper_try_add_edge!(self);

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        self.adj_out.try_remove_edge(u, v)
            && match self.adj_in.get_mut(&v).unwrap().remove(&u) {
                true => true,
                false => panic!("Edge not found in adj_in, should not happen!"),
            }
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node) {
        let mut edges = FxHashSet::default();
        std::mem::swap(&mut edges, self.adj_in.get_mut(&u).unwrap());
        for v in edges {
            // removing (v, u)
            self.adj_out.remove_edge(v, u);
        }
    }

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node) {
        let mut edges = FxHashSet::default();
        std::mem::swap(&mut edges, self.adj_out.adj.get_mut(&u).unwrap());
        self.adj_out.m -= edges.len();
        for v in edges {
            // removing (u, v)
            self.adj_in.get_mut(&v).unwrap().remove(&u);
        }
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        self.try_remove_edge(u, u);

        let in_neighbors = std::mem::take(self.adj_in.get_mut(&u).unwrap());
        let out_neighbors = std::mem::take(self.adj_out.adj.get_mut(&u).unwrap());
        self.adj_out.m -= in_neighbors.len() + out_neighbors.len();

        for &v in out_neighbors.iter() {
            self.adj_in.get_mut(&v).unwrap().remove(&u);
        }

        let mut loops = Vec::new();
        for v in in_neighbors {
            self.adj_out.adj.get_mut(&v).unwrap().remove(&u);

            for &w in &out_neighbors {
                self.try_add_edge(v, w);
                if v == w {
                    loops.push(v);
                }
            }
        }

        loops
    }
}

impl GraphNew for HashGraph {
    fn new(n: usize) -> Self {
        let mut adj: FxHashMap<Node, FxHashSet<Node>> =
            FxHashMap::with_capacity_and_hasher(n, Default::default());
        for v in 0..n {
            adj.insert(v as Node, FxHashSet::default());
        }
        Self { adj, m: 0 }
    }
}

impl GraphNew for HashGraphIn {
    fn new(n: usize) -> Self {
        let mut adj_in: FxHashMap<Node, FxHashSet<Node>> =
            FxHashMap::with_capacity_and_hasher(n, Default::default());
        for v in 0..n {
            adj_in.insert(v as Node, FxHashSet::default());
        }

        Self {
            adj_out: HashGraph::new(n),
            adj_in,
        }
    }
}

#[cfg(test)]
pub mod tests_hash_graph {
    use super::graph_macros::{base_tests, test_helper_vertex_editing};
    use super::*;
    base_tests!(HashGraph);
    test_helper_vertex_editing!(HashGraph);
}

#[cfg(test)]
pub mod tests_hash_graph_in {
    use super::graph_macros::{base_tests_in, test_helper_vertex_editing};
    use super::*;
    base_tests_in!(HashGraphIn);
    test_helper_vertex_editing!(HashGraphIn);
}
