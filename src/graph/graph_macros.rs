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

pub(super) use impl_helper_adjacency_list;
pub(super) use impl_helper_adjacency_test;
pub(super) use impl_helper_adjacency_test_linear_search_bi_directed;
pub(super) use impl_helper_graph_debug;
pub(super) use impl_helper_graph_from_edges;
pub(super) use impl_helper_graph_order;
