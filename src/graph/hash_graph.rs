use super::io::DotWrite;
use super::*;
use fxhash::{FxHashMap, FxHashSet};
use std::fmt::Debug;
use std::{fmt, str};

#[derive(Clone)]
pub struct HashGraph {
    adj: FxHashMap<Node, FxHashSet<Node>>,
    m: usize,
}

#[derive(Clone)]
pub struct HashGraphIn {
    adj_out: HashGraph,
    adj_in: FxHashMap<Node, FxHashSet<Node>>,
}

graph_macros::impl_helper_graph_debug!(HashGraph);
graph_macros::impl_helper_graph_from_edges!(HashGraph);

graph_macros::impl_helper_graph_debug!(HashGraphIn);
graph_macros::impl_helper_graph_from_edges!(HashGraphIn);
graph_macros::impl_helper_adjacency_list!(HashGraphIn, adj_out);
graph_macros::impl_helper_adjacency_test!(HashGraphIn, adj_out);
graph_macros::impl_helper_graph_order!(HashGraphIn, adj_out);

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

    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.adj.contains_key(&u));
        assert!(self.adj.get(&u).unwrap().contains(&v));
        self.adj.get_mut(&u).unwrap().remove(&v);
        self.m -= 1;
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

    fn remove_edge(&mut self, u: Node, v: Node) {
        self.adj_out.remove_edge(u, v);
        self.adj_in.get_mut(&v).unwrap().remove(&u);
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
}

impl GraphNew for HashGraph {
    fn new(n: usize) -> Self {
        Self {
            adj: FxHashMap::with_capacity_and_hasher(n, Default::default()),
            m: 0,
        }
    }
}

impl GraphNew for HashGraphIn {
    fn new(n: usize) -> Self {
        Self {
            adj_out: HashGraph::new(n),
            adj_in: FxHashMap::with_capacity_and_hasher(n, Default::default()),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn graph_edges() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = HashGraph::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn test_remove_edges() {
        let org_graph = HashGraph::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

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
        let mut g = HashGraph::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }

    #[test]
    fn test_remove_vertex() {
        let org_graph = HashGraph::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (5, 4)]);
        let mut graph = org_graph.clone();

        graph.remove_vertex(3);
        assert_eq!(graph.has_vertex(3), false);
        assert_eq!(graph.number_of_nodes(), org_graph.number_of_nodes() - 1);
        assert_eq!(graph.number_of_edges(), 1);
    }

    #[test]
    fn test_remove_vertex_in() {
        let org_graph = HashGraphIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (5, 4)]);
        let mut graph = org_graph.clone();

        graph.remove_vertex(3);
        assert_eq!(graph.has_vertex(3), false);
        assert_eq!(graph.number_of_nodes(), org_graph.number_of_nodes() - 1);
        assert_eq!(graph.number_of_edges(), 1);
    }

    #[test]
    fn graph_edges_in() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = HashGraphIn::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn test_remove_edges_in() {
        let org_graph = HashGraphIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

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
    fn test_debug_format_in() {
        let mut g = HashGraphIn::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }
}
