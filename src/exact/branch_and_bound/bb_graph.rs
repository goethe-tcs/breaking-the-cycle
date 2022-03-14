#![allow(unused_macros, unused_imports)]

use super::*;
use crate::utils::int_iterator::IntegerIterators;
use bitintr::Pext;
use num::cast::AsPrimitive;
use num::{FromPrimitive, One, PrimInt};
use std::ops::{BitOrAssign, Range, ShlAssign};

pub trait GraphItem:
    Copy
    + PrimInt
    + Pext
    + BitOrAssign
    + ShlAssign
    + FromPrimitive
    + IntegerIterators
    + AsPrimitive<u64>
    + AsPrimitive<u32>
    + AsPrimitive<u16>
    + AsPrimitive<u8>
{
    // This is already added to PrimInt but not published yet, see
    // https://github.com/rust-num/num-traits/pull/205/files
    fn trailing_ones(self) -> u32 {
        (!self).trailing_zeros()
    }
}

impl GraphItem for u8 {}
impl GraphItem for u16 {}
impl GraphItem for u32 {}
impl GraphItem for u64 {}

pub(super) trait BBGraph: Sized {
    type NodeMask: GraphItem;
    type SccIterator<'a>: Iterator<Item = Self::NodeMask>
    where
        Self: 'a;

    const CAPACITY: usize;

    fn from_bbgraph<G: BBGraph>(graph: &G) -> Self
    where
        <G as BBGraph>::NodeMask: num::traits::AsPrimitive<Self::NodeMask>;

    fn len(&self) -> usize;

    fn remove_first_node(&self) -> Self;
    fn remove_node(&self, u: Node) -> Self;
    fn contract_first_node(&self) -> Self;

    fn transitive_closure(&self) -> Self;

    fn nodes_with_loops(&self) -> Self::NodeMask;
    fn has_node_with_loop(&self) -> bool;

    fn has_all_edges(&self) -> bool;

    fn vertices(&self) -> Range<usize> {
        0..self.len()
    }

    fn sccs(&self) -> Self::SccIterator<'_>;
    fn subgraph(&self, included_nodes: Self::NodeMask) -> Self;

    /// Remove loops and return the loops, the mask of the resulting subgraph
    /// without loops, and the resulting subgraph.
    fn remove_loops(&self) -> (Self::NodeMask, Self::NodeMask, Self) {
        let loops = self.nodes_with_loops();
        let nodes_without_loops = !loops & self.nodes_mask();

        (
            loops,
            nodes_without_loops,
            self.subgraph(nodes_without_loops),
        )
    }

    fn first_node_has_loop(&self) -> bool;
    fn nodes_mask(&self) -> Self::NodeMask;

    fn out_neighbors(&self, u: Node) -> Self::NodeMask;

    /// Tests whether the edge (u,v) exists. This method is intended for debugging and may have
    /// not optimal performance
    fn has_edge(&self, u: Node, v: Node) -> bool {
        debug_assert!((v as usize) < Self::CAPACITY);
        (self.out_neighbors(u) >> (v as usize)) & Self::NodeMask::one() == Self::NodeMask::one()
    }

    /// Returns a vector containing all edges contained in the graph sorted in lexicographical
    /// order. This method is intended for debugging and may not have optimal performance.
    fn edges(&self) -> Vec<Edge> {
        self.vertices()
            .flat_map(|u| {
                self.out_neighbors(u as Node)
                    .iter_ones()
                    .map(move |v| (u as Node, v as Node))
            })
            .collect()
    }
}

macro_rules! bbgraph_tests {
    ($t:ty, $n:ident) => {
        paste::item! {
            #[test]
            fn [< from_$n >]() {
                let edges = [(0, 1), (1, 1), (2, 3), (6, 5)];
                let graph = $t::from(&AdjListMatrix::from(&edges));
                assert_eq!(graph.edges(), Vec::from(edges));
            }

            #[test]
            fn [< acyclic_$n  >] () {
                for i in 1_usize..$t::CAPACITY {
                    let mut adjmat = AdjListMatrix::new(i);
                    let path_vertices: Vec<Node> = adjmat.vertices().collect();
                    adjmat.connect_path(path_vertices.into_iter());
                    {
                        let graph = $t::from(&adjmat);
                        assert_eq!(graph.transitive_closure().nodes_with_loops(), 0, "i={}", i);
                    }
                    adjmat.add_edge(i as Node - 1, 0);
                    {
                        let graph = $t::from(&adjmat);
                        assert_eq!(
                            graph.transitive_closure().nodes_with_loops() as u64,
                            ((1u64 << i) - 1) as u64,
                            "i={}",
                            i
                        );
                    }
                }
            }

            #[test]
            fn [<remove_node_$n >]() {
                assert!($t::CAPACITY >= 8);
                let edges = [(0, 1), (1, 1), (2, 3), (6, 5)];
                let graph = $t::from(&AdjListMatrix::from(&edges));
                assert_eq!(graph.remove_node(0).edges(), vec![(0, 0), (1, 2), (5, 4)]);
                assert_eq!(graph.remove_node(1).edges(), vec![(1, 2), (5, 4)]);
                assert_eq!(graph.remove_node(2).edges(), vec![(0, 1), (1, 1), (5, 4)]);
                assert_eq!(graph.remove_node(3).edges(), vec![(0, 1), (1, 1), (5, 4)]);
                assert_eq!(
                    graph.remove_node(4).edges(),
                    vec![(0, 1), (1, 1), (2, 3), (5, 4)]
                );
                assert_eq!(graph.remove_node(5).edges(), vec![(0, 1), (1, 1), (2, 3)]);
                assert_eq!(graph.remove_node(6).edges(), vec![(0, 1), (1, 1), (2, 3)]);
                assert_eq!(graph.remove_node(6).len(), 6);

                assert_eq!(
                    graph.remove_node(0).remove_node(0).edges(),
                    vec![(0, 1), (4, 3)]
                );
                assert_eq!(graph.remove_node(3).remove_node(1).edges(), vec![(4, 3)]);
                assert_eq!(
                    graph.remove_node(2).remove_node(5).edges(),
                    vec![(0, 1), (1, 1)]
                );
                assert_eq!(
                    graph.remove_node(4).remove_node(0).edges(),
                    vec![(0, 0), (1, 2), (4, 3)]
                );
                assert_eq!(
                    graph.remove_node(5).remove_node(2).edges(),
                    vec![(0, 1), (1, 1)]
                );
            }

            #[test]
            fn [<subgraph_$n >]() {
                assert!($t::CAPACITY >= 8);
                let edges = [(0, 1), (0, 4), (1, 1), (2, 3), (6, 5), (6, 7), (7, 5)];
                let graph = $t::from(&AdjListMatrix::from(&edges));
                let subgraph = graph.subgraph(0);
                assert_eq!(subgraph.len(), 0);
                assert!(subgraph.edges().is_empty());

                let subgraph = graph.subgraph(0b00000011);
                assert_eq!(subgraph.len(), 2);
                assert_eq!(
                    subgraph.edges(),
                    vec![(0, 1), (1, 1)]
                );

                let subgraph = graph.subgraph(0b00010111);
                assert_eq!(subgraph.len(), 4);
                assert_eq!(
                    subgraph.edges(),
                    vec![(0, 1), (0, 3), (1, 1)]
                );

                let subgraph = graph.subgraph(0b11101000);
                assert_eq!(subgraph.len(), 4);
                assert_eq!(
                    subgraph.edges(),
                    vec![(2, 1), (2, 3), (3, 1)]
                );

                let subgraph = graph
                    .remove_first_node()
                    .remove_first_node()
                    .subgraph(0b00111111);
                assert_eq!(subgraph.len(), 6);
                assert_eq!(
                    subgraph.edges(),
                    vec![(0, 1), (4, 3), (4, 5), (5, 3)]
                );

                let subgraph = graph.remove_node(2).remove_node(3).subgraph(0b00111111);
                assert_eq!(subgraph.len(), 6);
                assert_eq!(
                    subgraph.edges(),
                    vec![
                        (0, 1),
                        (1, 1),
                        (4, 3),
                        (4, 5),
                        (5, 3)
                    ]
                );

                let subgraph = graph
                    .remove_node(0)
                    .remove_node(6)
                    .remove_node(2)
                    .subgraph(0b00011001);
                assert_eq!(subgraph.len(), 3);
                assert_eq!(
                    subgraph.edges(),
                    vec![(0, 0), (2, 1)]
                );
            }

            #[test]
            fn [<remove_loops_$n >]() {
                let graph = |edges| $t::from(&AdjListMatrix::from(&edges));

                let remove_loops_test = |graph: $t,
                                         graph_without_loops,
                                         graph_without_loops_mask,
                                         loops_mask| {
                    assert_eq!(
                        graph.remove_loops(),
                        (loops_mask, graph_without_loops_mask, graph_without_loops)
                    );
                };

                remove_loops_test(graph(vec![]), graph(vec![]), 0, 0);
                remove_loops_test(graph(vec![(0, 1)]), graph(vec![(0, 1)]), 0b00000011, 0);
                remove_loops_test(graph(vec![(0, 0)]), graph(vec![]), 0, 0b00000001);
                remove_loops_test(graph(vec![(0, 0)]), graph(vec![]), 0, 0b00000001);
                remove_loops_test(
                    graph(vec![(0, 0), (1, 1), (2, 2)]),
                    graph(vec![]),
                    0,
                    0b00000111,
                );
                remove_loops_test(
                    graph(vec![(0, 0), (1, 0), (2, 2)]),
                    graph(vec![(0, 0), (1, 0), (2, 2)])
                        .remove_node(0)
                        .remove_node(1),
                    0b00000010,
                    0b00000101,
                );
                remove_loops_test(
                    graph(vec![(3, 4), (1, 0), (2, 5), (3, 3), (4, 1)]),
                    graph(vec![(3, 4), (1, 0), (2, 5), (3, 3), (4, 1)]).remove_node(3),
                    0b00110111,
                    0b00001000,
                );
                remove_loops_test(
                    graph(vec![(3, 7), (7, 7), (0, 0), (1, 2), (2, 1)]),
                    graph(vec![(3, 7), (7, 7), (0, 0), (1, 2), (2, 1)])
                        .remove_node(0)
                        .remove_node(6),
                    0b01111110,
                    0b10000001,
                );
            }

            #[test]
            fn [< contract_$n >]() {
                let graph = $t::from(&AdjListMatrix::from(&[(1,0), (2, 0), (0, 2), (0, 3), (0,4), (3,4)]));
                let contr = graph.contract_first_node();
                assert_eq!(contr.edges(), vec![(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,3)]);
            }

            #[test]
            fn [< diagonals_$n >]() {
                let rng = &mut rand::thread_rng();

                for n in 1..$t::CAPACITY {
                    let mut org_graph : AdjListMatrix = crate::random_models::gnp::generate_gnp(rng, $t::CAPACITY as Node, 0.2);
                    for i in org_graph.vertices_range() {
                        org_graph.try_remove_edge(i, i);
                    }

                    // no diag
                    {
                        let graph = $t::from(&org_graph);
                        assert!(!graph.has_node_with_loop());
                        assert_eq!(graph.nodes_with_loops(), 0);
                    }

                    for u in 0..n {
                        let mut graph = org_graph.clone();
                        graph.add_edge(u as Node, u as Node);

                        let graph = $t::from(&graph);
                        assert!(graph.has_node_with_loop());
                        assert_eq!(graph.nodes_with_loops() >> u, 1);
                    }
                }
            }

            #[test]
            fn [< sccs_$n >]() {
                let scc_vec = |edges| {
                    $t::from(&AdjListMatrix::from(&edges))
                        .transitive_closure()
                        .sccs()
                        .map(|scc| {
                            assert!(scc as u64 <= 255); // Only graphs with up to 8 nodes to simplify tests
                            scc as u8
                        })
                        .collect_vec()
                };

                assert_eq!(scc_vec(vec![]), vec![]);
                assert_eq!(scc_vec(vec![(0, 0)]), vec![0b00000001]);
                assert_eq!(scc_vec(vec![(0, 1)]), vec![0b00000001, 0b00000010]);
                assert_eq!(
                    scc_vec(vec![(0, 3)]),
                    vec![0b00000001, 0b00000010, 0b00000100, 0b00001000]
                );
                assert_eq!(scc_vec(vec![(0, 1), (1, 0)]), vec![0b00000011]);
                assert_eq!(
                    scc_vec(vec![(0, 1), (1, 1), (1, 2)]),
                    vec![0b00000001, 0b00000010, 0b00000100]
                );
                assert_eq!(
                    scc_vec(vec![(0, 1), (1, 1), (1, 2), (2, 0)]),
                    vec![0b00000111]
                );
                assert_eq!(
                    scc_vec(vec![(0, 1), (1, 1), (1, 2), (2, 1)]),
                    vec![0b00000001, 0b00000110]
                );
                assert_eq!(
                    scc_vec(vec![
                        (0, 1),
                        (1, 1),
                        (1, 2),
                        (2, 1),
                        (2, 4),
                        (4, 3),
                        (3, 7),
                        (7, 2),
                    ]),
                    vec![0b00000001, 0b10011110, 0b00100000, 0b01000000]
                );
                assert_eq!(
                    scc_vec(vec![(0, 1), (1, 2), (2, 3), (3, 0), (5, 6)]),
                    vec![0b00001111, 0b00010000, 0b00100000, 0b01000000]
                );
            }
        }
    };
}

pub(super) use bbgraph_tests;
