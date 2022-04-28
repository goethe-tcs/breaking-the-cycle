// cargo check does not detect imports and macro use in test modules. This is the only workaround
#![allow(unused_macros, unused_imports)]

macro_rules! impl_helper_graph_default {
    ($t:ident) => {
        impl Default for $t {
            fn default() -> Self {
                $t::new(0)
            }
        }
    };
}

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

macro_rules! impl_helper_graph_from_slice {
    ($t:ident) => {
        impl GraphFromSlice for $t {
            fn from_slice(n: Node, edges: &[(Node, Node)], known_unique: bool) -> Self {
                let mut graph = Self::new(n as usize);
                for e in edges {
                    if known_unique {
                        graph.add_edge(e.0, e.1);
                    } else {
                        graph.try_add_edge(e.0, e.1);
                    }
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
                    f.write_str(str::from_utf8(&buf).unwrap().trim())?;
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

macro_rules! impl_helper_undir_adjacency_test {
    ($t:ident) => {
        impl AdjacencyTestUndir for $t {
            fn has_undir_edge(&self, u: Node, v: Node) -> bool {
                self.has_edge(u, v) && self.has_edge(v, u)
            }
        }
    };
}

macro_rules! impl_helper_adjacency_list_undir {
    ($t:ident) => {
        impl AdjacencyListUndir for $t {
            type IterUndir<'a> = impl Iterator<Item = Node> + 'a;
            type IterOutOnly<'a> = impl Iterator<Item = Node> + 'a;
            type IterInOnly<'a> = impl Iterator<Item = Node> + 'a;

            fn undir_neighbors(&self, u: Node) -> Self::IterUndir<'_> {
                let hash_out = FxHashSet::from_iter(self.out_neighbors(u));
                let hash_in = FxHashSet::from_iter(self.in_neighbors(u));
                let intersection: Vec<Node> = hash_out.intersection(&hash_in).copied().collect();
                intersection.into_iter()
            }

            fn out_only_neighbors(&self, u: Node) -> Self::IterOutOnly<'_> {
                let hash_in = FxHashSet::from_iter(self.in_neighbors(u));
                self.out_neighbors(u).filter(move |v| !hash_in.contains(v))
            }

            fn in_only_neighbors(&self, u: Node) -> Self::IterInOnly<'_> {
                let hash_out = FxHashSet::from_iter(self.out_neighbors(u));
                self.in_neighbors(u).filter(move |v| !hash_out.contains(v))
            }

            fn undir_degree(&self, u: Node) -> Node {
                let hash_out = FxHashSet::from_iter(self.out_neighbors(u));
                let hash_in = FxHashSet::from_iter(self.in_neighbors(u));
                hash_out.intersection(&hash_in).count() as Node
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
            edges.sort();
            assert_eq!(edges, graph.edges_vec());
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
            assert_eq!(graph.edges_vec(), edges);
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
            // has_edge
            {
                let edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)];
                let graph = $t::from(&edges);
                for (u, v) in edges {
                    assert!(graph.has_edge(u, v));
                    assert!(!graph.has_edge(v, u));
                }
            }

            // has_self_loop
            {
                let edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5)];
                let graph = $t::from(&edges);
                assert!(!graph.has_self_loop());

                let edges = vec![(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (3, 3)];
                let graph = $t::from(&edges);
                assert!(graph.has_self_loop());
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
                assert_eq!(graph.edges_vec(), org_graph.edges_vec());

                graph.remove_edges_out_of_node(4);
                assert_eq!(graph.edges_vec(), org_graph.edges_vec());
            }

            // remove out
            {
                let mut graph = org_graph.clone();

                graph.remove_edges_out_of_node(3);
                assert_eq!(
                    graph.number_of_edges(),
                    org_graph.number_of_edges() - org_graph.out_degree(3) as usize
                );
                for (u, _v) in graph.edges_iter() {
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
                for (_u, v) in graph.edges_iter() {
                    assert_ne!(v, 3);
                }
            }

            // remove in- and out
            {
                let mut graph = org_graph.clone();

                let num_edges_to_delete = graph
                    .vertices()
                    .filter(|v| graph.has_edge(*v, 3) || graph.has_edge(3, *v))
                    .count();

                graph.remove_edges_at_node(3);
                assert_eq!(
                    graph.number_of_edges(),
                    org_graph.number_of_edges() - num_edges_to_delete
                );
                for (u, v) in graph.edges_iter() {
                    assert_ne!(v, 3);
                    assert_ne!(u, 3);
                }
            }
        }

        #[test]
        fn contract_node() {
            // path
            {
                let mut org_graph = $t::from(&[(0, 1), (1, 2)]);
                org_graph.contract_node(1);
                assert_eq!(org_graph.edges_vec(), vec![(0, 2)]);
                assert_eq!(org_graph.number_of_edges(), 1);
            }

            // cross
            {
                let mut org_graph = $t::from(&[(0, 2), (1, 2), (2, 3), (2, 4)]);
                org_graph.contract_node(2);
                assert_eq!(org_graph.edges_vec(), vec![(0, 3), (0, 4), (1, 3), (1, 4)]);
                assert_eq!(org_graph.number_of_edges(), 4);
            }

            // cross with self-loop
            {
                let mut org_graph = $t::from(&[(0, 2), (1, 2), (2, 3), (2, 4), (2, 2)]);
                org_graph.contract_node(2);
                assert_eq!(org_graph.edges_vec(), vec![(0, 3), (0, 4), (1, 3), (1, 4)]);
                assert_eq!(org_graph.number_of_edges(), 4);
            }

            // loop
            {
                let mut org_graph = $t::from(&[(0, 1), (1, 0)]);
                let loops = org_graph.contract_node(1);
                assert_eq!(org_graph.edges_vec(), vec![(0, 0)]);
                assert_eq!(org_graph.number_of_edges(), 1);
                assert_eq!(loops, vec![0]);
            }

            // no in-neighbors
            {
                let mut org_graph = $t::from(&[(0, 1), (1, 2)]);
                org_graph.contract_node(0);
                assert_eq!(org_graph.edges_vec(), vec![(1, 2)]);
                assert_eq!(org_graph.number_of_edges(), 1);
            }

            // no out-neighbors
            {
                let mut org_graph = $t::from(&[(0, 1), (1, 2)]);
                org_graph.contract_node(2);
                assert_eq!(org_graph.edges_vec(), vec![(0, 1)]);
                assert_eq!(org_graph.number_of_edges(), 1);
            }
        }
    };
}

macro_rules! test_helper_default {
    ($t:ident) => {
        #[test]
        fn test_default() {
            let graph = $t::default();
            assert_eq!(graph.len(), 0);
            assert_eq!(graph.number_of_edges(), 0);
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

macro_rules! test_helper_undir {
    ($t:ident) => {
        #[test]
        fn undir_neighbors() {
            use itertools::Itertools;
            use rand::prelude::*;

            let mut rng = rand_pcg::Pcg64::seed_from_u64(1234);
            let mut edges = Vec::from([(0, 1), (0, 2), (0, 3), (1, 0), (3, 0), (4, 0)]);
            for _ in 0..10 {
                edges.shuffle(&mut rng);
                let graph = $t::from(edges.as_slice());

                fn check_sorted(mut vec: Vec<Node>, expected: Vec<Node>) {
                    vec.sort_unstable();
                    assert_eq!(vec, expected);
                }

                check_sorted(graph.out_neighbors(0).collect_vec(), vec![1, 2, 3]);
                check_sorted(graph.in_neighbors(0).collect_vec(), vec![1, 3, 4]);
                check_sorted(graph.undir_neighbors(0).collect_vec(), vec![1, 3]);
            }
        }

        #[test]
        fn dynamic_undir_neighbors() {
            use itertools::Itertools;
            use rand::SeedableRng;
            use rand::*;
            use std::collections::HashSet;

            let mut rng = rand_pcg::Pcg64::seed_from_u64(1234);

            for n in [1, 2, 5, 10] {
                let mut edges = HashSet::with_capacity((n * n) as usize);
                let mut graph = $t::new(n as usize);

                for _ in 0..(5 * n * n) {
                    let u = rng.gen_range(0..n);
                    let v = rng.gen_range(0..n);

                    if edges.insert((u, v)) {
                        graph.add_edge(u, v);
                    } else {
                        edges.remove(&(u, v));
                        graph.remove_edge(u, v);
                    }

                    for u in 0..n {
                        let out_n = (0..n)
                            .into_iter()
                            .filter(|&v| edges.contains(&(u, v)))
                            .collect_vec();
                        let in_n = (0..n)
                            .into_iter()
                            .filter(|&v| edges.contains(&(v, u)))
                            .collect_vec();
                        let undir_n = (0..n)
                            .into_iter()
                            .filter(|&v| edges.contains(&(u, v)) && edges.contains(&(v, u)))
                            .collect_vec();

                        assert_eq!(graph.out_degree(u) as usize, out_n.len());
                        assert_eq!(graph.in_degree(u) as usize, in_n.len());
                        assert_eq!(graph.undir_degree(u) as usize, undir_n.len());

                        let mut g_out = graph.out_neighbors(u).collect_vec();
                        let mut g_in = graph.in_neighbors(u).collect_vec();
                        let mut g_undir = graph.undir_neighbors(u).collect_vec();

                        g_out.sort_unstable();
                        g_in.sort_unstable();
                        g_undir.sort_unstable();

                        assert_eq!(g_out, out_n);
                        assert_eq!(g_in, in_n);
                        assert_eq!(g_undir, undir_n);
                    }

                    for u in 0..n {
                        for v in 0..n {
                            assert_eq!(graph.has_edge(u, v), edges.contains(&(u, v)));
                        }
                    }

                    assert_eq!(graph.number_of_edges(), edges.len());
                }
            }
        }
    };
}

macro_rules! test_helper_from_slice {
    ($t:ident) => {
        #[test]
        fn from_slice() {
            use rand::prelude::*;

            let edges = [(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (5, 4)];
            let org_graph = $t::from(&edges);
            {
                let from_slice_graph = $t::from_slice(6, &edges, true);
                assert_eq!(org_graph.edges_vec(), from_slice_graph.edges_vec());
            }

            let mut rng = rand_pcg::Pcg64::seed_from_u64(1234);
            let mut double_edges = Vec::from(edges);
            double_edges.extend(edges);

            for _ in 0..5 {
                double_edges.shuffle(&mut rng);
                let from_slice_graph = $t::from_slice(6, &edges, false);
                assert_eq!(org_graph.edges_vec(), from_slice_graph.edges_vec());
            }
        }
    };
}

macro_rules! base_tests {
    ($t:ident) => {
        test_helper_default!($t);
        test_helper_from_edges!($t);
        test_helper_graph_order!($t);
        test_helper_adjacency_list!($t);
        test_helper_adjacency_test!($t);
        test_helper_graph_edge_editing!($t);
        test_helper_debug_format!($t);
        test_helper_from_slice!($t);
    };
}

macro_rules! base_tests_in {
    ($t:ident) => {
        base_tests!($t);
        test_helper_adjacency_list_in!($t);
        test_helper_undir!($t);
    };
}

pub(super) use impl_helper_adjacency_list;
pub(super) use impl_helper_adjacency_list_undir;
pub(super) use impl_helper_adjacency_test;
pub(super) use impl_helper_adjacency_test_linear_search_bi_directed;
pub(super) use impl_helper_graph_debug;
pub(super) use impl_helper_graph_default;
pub(super) use impl_helper_graph_from_edges;
pub(super) use impl_helper_graph_from_slice;
pub(super) use impl_helper_graph_order;
pub(super) use impl_helper_try_add_edge;
pub(super) use impl_helper_undir_adjacency_test;

pub(super) use base_tests;
pub(super) use base_tests_in;
pub(super) use test_helper_adjacency_list;

pub(super) use test_helper_adjacency_list_in;
pub(super) use test_helper_adjacency_test;
pub(super) use test_helper_debug_format;
pub(super) use test_helper_default;
pub(super) use test_helper_from_edges;
pub(super) use test_helper_from_slice;
pub(super) use test_helper_graph_edge_editing;
pub(super) use test_helper_graph_order;
pub(super) use test_helper_undir;
pub(super) use test_helper_vertex_editing;
