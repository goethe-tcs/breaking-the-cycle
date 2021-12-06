// cargo check does not detect imports and macro use in test modules. This is the only workaround
#![allow(unused_macros, unused_imports)]

macro_rules! impl_helper_graph_from_edges {
    ($t:ident) => {
        impl<'a, T: IntoIterator<Item = &'a Edge> + Clone> From<T> for $t {
            fn from(edges: T) -> Self {
                let n = edges
                    .clone()
                    .into_iter()
                    .map(|e| e.0.max(e.1) + 1)
                    .max()
                    .unwrap_or(0);
                let mut graph = Self::new(n as usize);
                for e in edges {
                    graph.add_edge(e.0, e.1);
                }
                graph
            }
        }
    };
}

macro_rules! impl_helper_graph_debug {
    ($t:ident) => {
        impl Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut buf = Vec::new();
                if self.try_write_dot(&mut buf).is_ok() {
                    f.write_str(str::from_utf8(&buf).unwrap())?;
                }

                Ok(())
            }
        }
    };
}

macro_rules! impl_helper_graph_order {
    ($t:ident, $field:ident) => {
        impl GraphOrder for $t {
            type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

            fn number_of_nodes(&self) -> Node {
                self.$field.number_of_nodes()
            }
            fn number_of_edges(&self) -> usize {
                self.$field.number_of_edges()
            }

            fn vertices(&self) -> Self::VertexIter<'_> {
                self.$field.vertices()
            }
        }
    };
}

macro_rules! impl_helper_adjacency_list {
    ($t:ident, $field:ident) => {
        impl AdjacencyList for $t {
            type Iter<'a> = impl Iterator<Item = Node> + 'a;

            fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
                self.$field.out_neighbors(u)
            }

            fn out_degree(&self, u: Node) -> Node {
                self.$field.out_degree(u)
            }
        }
    };
}

macro_rules! impl_helper_adjacency_test {
    ($t:ident, $field:ident) => {
        impl AdjacencyTest for $t {
            fn has_edge(&self, u: Node, v: Node) -> bool {
                self.$field.has_edge(u, v)
            }
        }
    };
}

macro_rules! impl_helper_adjacency_test_linear_search_bi_directed {
    ($t:ident) => {
        impl AdjacencyTest for $t {
            fn has_edge(&self, u: Node, v: Node) -> bool {
                if self.in_degree(v) < self.out_degree(u) {
                    self.in_neighbors(v).any(|w| w == u)
                } else {
                    self.out_neighbors(u).any(|w| w == v)
                }
            }
        }
    };
}

macro_rules! impl_helper_try_add_edge {
    ($self:ident) => {
        fn try_add_edge(&mut $self, u: Node, v: Node) -> bool {
            if $self.has_edge(u, v) {
                false
            } else {
                $self.add_edge(u, v);
                true
            }
        }
    };
}

macro_rules! test_helper_from_edges {
    ($t:ident) => {
        #[test]
        fn graph_edges() {
            let mut edges = vec![(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)];
            let graph = $t::from(&edges);
            assert_eq!(graph.number_of_nodes(), 6);
            assert_eq!(graph.number_of_edges(), edges.len());
            let mut ret_edges = graph.edges();

            edges.sort();
            ret_edges.sort();

            assert_eq!(edges, ret_edges);
        }
    };
}

macro_rules! test_helper_graph_order {
    ($t:ident) => {
        #[test]
        fn graph_order() {
            let graph = $t::new(10);
            assert_eq!(graph.number_of_nodes(), 10);
            assert_eq!(graph.number_of_nodes(), graph.len() as Node);
            assert_eq!(graph.number_of_edges(), 0);

            let mut vertices: Vec<_> = graph.vertices().collect();
            vertices.sort();
            let expected: Vec<_> = (0..10).collect();
            assert_eq!(vertices, expected);

            assert!(!graph.is_empty());
            let graph = $t::new(0);
            assert!(graph.is_empty());
            assert_eq!(graph.number_of_nodes(), 0);
            assert_eq!(graph.number_of_edges(), 0);
            assert_eq!(graph.number_of_nodes(), graph.len() as Node);
        }
    };
}

macro_rules! test_helper_adjacency_list {
    ($t:ident) => {
        #[test]
        fn adjacency_list() {
            let mut edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)];
            edges.sort();
            let graph = $t::from(&edges);

            let mut nb: Vec<_> = graph.out_neighbors(3).collect();
            nb.sort();
            let expected: Vec<_> = vec![4, 5];
            assert_eq!(nb, expected);

            assert_eq!(nb.len() as Node, graph.out_degree(3));

            let mut tmp = graph.edges();
            tmp.sort();
            assert_eq!(tmp, edges);
        }
    };
}

macro_rules! test_helper_adjacency_list_in {
    ($t:ident) => {
        #[test]
        fn adjacency_list_in() {
            let mut edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)];
            edges.sort();
            let graph = $t::from(&edges);

            let mut nb: Vec<_> = graph.in_neighbors(3).collect();
            nb.sort();
            let expected: Vec<_> = vec![0, 1, 2];
            assert_eq!(nb, expected);

            assert_eq!(3, graph.in_degree(3));
            assert_eq!(5, graph.total_degree(3));
        }
    };
}

macro_rules! test_helper_adjacency_test {
    ($t:ident) => {
        #[test]
        fn adjacency_test() {
            let edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)];
            let graph = $t::from(&edges);
            for (u, v) in edges {
                assert!(graph.has_edge(u, v));
                assert!(!graph.has_edge(v, u));
            }
        }
    };
}

macro_rules! test_helper_debug_format {
    ($t:ident) => {
        #[test]
        fn test_debug_format_in() {
            let mut g = $t::new(8);
            g.add_edges(&[(0, 1), (0, 2), (0, 3), (4, 5)]);
            let str = format!("{:?}", g);
            assert!(str.contains("digraph"));
            assert!(str.contains("v0 ->"));
            assert!(!str.contains("v3 ->"));
        }
    };
}

macro_rules! test_helper_graph_edge_editing {
    ($t:ident) => {
        #[test]
        #[should_panic]
        fn edge_editing_panic() {
            let mut org_graph = $t::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);
            org_graph.remove_edge(3, 0);
        }

        #[test]
        fn edge_editing() {
            let org_graph = $t::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)]);

            // add_edge and try_add_edge
            {
                let mut graph = org_graph.clone();

                graph.add_edge(3, 0);
                assert!(graph.has_edge(3, 0));
                assert_eq!(org_graph.number_of_edges() + 1, graph.number_of_edges());

                graph.try_add_edge(3, 0);
                assert!(graph.has_edge(3, 0));
                assert_eq!(org_graph.number_of_edges() + 1, graph.number_of_edges());

                graph.try_add_edge(1, 5);
                assert!(graph.has_edge(1, 5));
                assert_eq!(org_graph.number_of_edges() + 2, graph.number_of_edges());
            }

            // remove_edge and try_remove_edge
            {
                let mut graph = org_graph.clone();

                graph.remove_edge(0, 3);
                assert!(!graph.has_edge(0, 3));
                assert_eq!(org_graph.number_of_edges() - 1, graph.number_of_edges());

                graph.try_remove_edge(0, 3);
                assert!(!graph.has_edge(3, 0));
                assert_eq!(org_graph.number_of_edges() - 1, graph.number_of_edges());

                graph.try_remove_edge(1, 3);
                assert!(!graph.has_edge(1, 3));
                assert_eq!(org_graph.number_of_edges() - 2, graph.number_of_edges());
            }

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
    };
}

macro_rules! test_helper_vertex_editing {
    ($t:ident) => {
        #[test]
        fn test_remove_vertex() {
            let org_graph = $t::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (5, 4)]);
            let mut graph = org_graph.clone();

            assert_eq!(graph.has_vertex(3), true);
            graph.remove_vertex(3);
            assert_eq!(graph.has_vertex(3), false);
            assert!(graph.has_edge(5, 4));
            assert_eq!(graph.number_of_nodes(), org_graph.number_of_nodes() - 1);
            assert_eq!(graph.number_of_edges(), 1);
        }
    };
}

macro_rules! base_tests {
    ($t:ident) => {
        test_helper_from_edges!($t);
        test_helper_graph_order!($t);
        test_helper_adjacency_list!($t);
        test_helper_adjacency_test!($t);
        test_helper_graph_edge_editing!($t);
        test_helper_debug_format!($t);
    };
}

macro_rules! base_tests_in {
    ($t:ident) => {
        base_tests!($t);
        test_helper_adjacency_list_in!($t);
    };
}

pub(super) use impl_helper_adjacency_list;
pub(super) use impl_helper_adjacency_test;
pub(super) use impl_helper_adjacency_test_linear_search_bi_directed;
pub(super) use impl_helper_graph_debug;
pub(super) use impl_helper_graph_from_edges;
pub(super) use impl_helper_graph_order;
pub(super) use impl_helper_try_add_edge;

pub(super) use base_tests;
pub(super) use base_tests_in;
pub(super) use test_helper_adjacency_list;
pub(super) use test_helper_adjacency_list_in;
pub(super) use test_helper_adjacency_test;
pub(super) use test_helper_debug_format;
pub(super) use test_helper_from_edges;
pub(super) use test_helper_graph_edge_editing;
pub(super) use test_helper_graph_order;
pub(super) use test_helper_vertex_editing;
