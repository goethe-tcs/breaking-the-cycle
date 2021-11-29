use super::io::DotWrite;
use super::*;
use std::fmt::Debug;
use std::{fmt, str};

/// A data structure for a directed graph supporting self-loops and multi-edges
#[derive(Clone)]
pub struct AdjArray {
    n: usize,
    m: usize,
    pub(super) out_neighbors: Vec<Vec<Node>>,
}

/// Same as AdjArray, but stores all in-edges for each vertex in addition to the outedges
#[derive(Clone)]
pub struct AdjArrayIn {
    in_neighbors: Vec<Vec<Node>>,
    adj: AdjArray,
}

graph_macros::impl_helper_graph_debug!(AdjArray);
graph_macros::impl_helper_graph_from_edges!(AdjArray);

graph_macros::impl_helper_graph_order!(AdjArrayIn, adj);
graph_macros::impl_helper_graph_debug!(AdjArrayIn);
graph_macros::impl_helper_graph_from_edges!(AdjArrayIn);
graph_macros::impl_helper_adjacency_list!(AdjArrayIn, adj);
graph_macros::impl_helper_adjacency_test_linear_search_bi_directed!(AdjArrayIn);

impl GraphOrder for AdjArray {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }
    fn number_of_edges(&self) -> usize {
        self.m
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        0..self.number_of_nodes()
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
        for v in 0..self.number_of_nodes() {
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
}

impl GraphEdgeEditing for AdjArrayIn {
    fn add_edge(&mut self, u: Node, v: Node) {
        self.adj.add_edge(u, v);
        self.in_neighbors[v as usize].push(u);
    }

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
pub mod tests {
    use super::*;

    #[test]
    fn graph_edges() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = AdjArray::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn try_remove_edge() {
        let mut org_graph = AdjArray::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        let old_m = org_graph.number_of_edges();
        assert!(org_graph.has_edge(0, 3));
        assert!(org_graph.try_remove_edge(0, 3));
        assert_eq!(org_graph.number_of_edges(), old_m - 1);
        assert!(!org_graph.has_edge(0, 3));

        let old_m = org_graph.number_of_edges();
        assert!(!org_graph.try_remove_edge(0, 3));
        assert_eq!(org_graph.number_of_edges(), old_m);
    }

    #[test]
    fn test_remove_edges() {
        let org_graph = AdjArray::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

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
        let mut g = AdjArray::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }

    #[test]
    fn try_remove_edge_in() {
        let mut org_graph = AdjArrayIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        let old_m = org_graph.number_of_edges();
        assert!(org_graph.has_edge(0, 3));
        assert!(org_graph.try_remove_edge(0, 3));
        assert_eq!(org_graph.number_of_edges(), old_m - 1);
        assert!(!org_graph.has_edge(0, 3));

        let old_m = org_graph.number_of_edges();
        assert!(!org_graph.try_remove_edge(0, 3));
        assert_eq!(org_graph.number_of_edges(), old_m);
    }

    #[test]
    #[should_panic]
    fn remove_edge_panic() {
        let mut org_graph = AdjArray::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        org_graph.remove_edge(3, 0);
    }

    #[test]
    #[should_panic]
    fn remove_edge_panic_in() {
        let mut org_graph = AdjArrayIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        org_graph.remove_edge(3, 0);
    }

    #[test]
    fn remove_edge() {
        let mut org_graph = AdjArray::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        let old_m = org_graph.number_of_edges();
        assert!(org_graph.has_edge(0, 3));
        org_graph.remove_edge(0, 3);
        assert_eq!(org_graph.number_of_edges(), old_m - 1);
        assert!(!org_graph.has_edge(0, 3));
    }

    #[test]
    fn remove_edge_in() {
        let mut org_graph = AdjArrayIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
        let old_m = org_graph.number_of_edges();
        assert!(org_graph.has_edge(0, 3));
        org_graph.remove_edge(0, 3);
        assert_eq!(org_graph.number_of_edges(), old_m - 1);
        assert!(!org_graph.has_edge(0, 3));
    }

    #[test]
    fn graph_edges_in() {
        let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
        let graph = AdjArrayIn::from(&edges);
        assert_eq!(graph.number_of_nodes(), 6);
        assert_eq!(graph.number_of_edges(), edges.len());
        let mut ret_edges = graph.edges();

        edges.sort();
        ret_edges.sort();

        assert_eq!(edges, ret_edges);
    }

    #[test]
    fn test_remove_edges_in() {
        let org_graph = AdjArrayIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

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
        let mut g = AdjArrayIn::new(8);
        g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
        let str = format!("{:?}", g);
        assert!(str.contains("digraph"));
        assert!(str.contains("v0 ->"));
        assert!(!str.contains("v3 ->"));
    }
}
