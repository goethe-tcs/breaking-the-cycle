#![allow(unused_macros, unused_imports)]

use super::*;
use crate::utils::*;
use bitintr::{Pdep, Pext};
use num::cast::AsPrimitive;
use num::{FromPrimitive, One, PrimInt};
use std::ops::{BitOrAssign, Range, ShlAssign};

pub trait GraphItem:
    Copy
    + PrimInt
    + BitOrAssign
    + ShlAssign
    + FromPrimitive
    + IntegerIterators
    + AsPrimitive<u128>
    + AsPrimitive<u64>
    + AsPrimitive<u32>
    + AsPrimitive<u16>
    + AsPrimitive<u8>
    + BitManip
{
}

impl GraphItem for u8 {}
impl GraphItem for u16 {}
impl GraphItem for u32 {}
impl GraphItem for u64 {}
impl GraphItem for u128 {}

pub trait BBGraph: Sized {
    type NodeMask: GraphItem;
    type SccIterator<'a>: Iterator<Item = Self::NodeMask>
    where
        Self: 'a;

    const CAPACITY: usize;

    fn from_bbgraph<G: BBGraph>(graph: &G) -> Self
    where
        <G as BBGraph>::NodeMask: num::traits::AsPrimitive<Self::NodeMask>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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

    fn remove_edges_at_nodes(&self, delete: Self::NodeMask) -> Self;

    fn first_node_has_loop(&self) -> bool;
    fn nodes_mask(&self) -> Self::NodeMask;
    fn out_neighbors(&self, u: Node) -> Self::NodeMask;

    /// Returns true iff there is no loop at node u and the node has
    /// either at most one out-neighbors or at most one in-neighbors.
    fn is_chain_node(&self, u: Node) -> bool {
        if (self.out_neighbors(u) >> (u as usize)) & Self::NodeMask::one() == Self::NodeMask::one()
        {
            return false;
        }

        if self.out_neighbors(u).count_ones() < 2 {
            return true;
        }

        let mut in_degree: u32 = 0;
        for v in self.vertices() {
            let contrib: u32 =
                ((self.out_neighbors(v as Node) >> (u as usize)) & Self::NodeMask::one()).as_();
            in_degree += contrib;
            if in_degree > 1 {
                return false;
            }
        }

        true
    }

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

    fn contract_chaining_nodes(&self) -> Self;

    fn node_with_most_undirected_edges(&self) -> Option<Node>;

    fn swap_nodes(&self, n0: Node, n1: Node) -> Self;

    fn node_with_max_out_degree(&self) -> Option<Node> {
        self.vertices()
            .map(|u| u as Node)
            .max_by_key(|&u| self.out_neighbors(u).count_ones())
    }
}

pub trait BBTryCompact<T>: BBGraph
where
    T: BBGraph,
{
    fn try_compact<I: FnMut(&T) -> Option<<T as BBGraph>::NodeMask>>(
        &self,
        callback: I,
    ) -> Option<Option<Self::NodeMask>>;
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
                    adjmat.connect_path(path_vertices.into_iter().rev());
                    {
                        let graph = $t::from(&adjmat);
                        assert_eq!(graph.transitive_closure().nodes_with_loops(), 0, "i={}", i);
                    }
                    adjmat.add_edge(0, i as Node - 1);
                    {
                        let graph = $t::from(&adjmat);
                        assert_eq!(
                            graph.transitive_closure().nodes_with_loops() as u128,
                            ((1u128 << i) - 1) as u128,
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
                use rand::Rng;
                let rng = &mut rand::thread_rng();

                for n in 1..$t::CAPACITY {
                    if n != $t::CAPACITY && n>8 && rng.gen_bool(0.5) {
                        continue;
                    }

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
                        if u != n && u>4 && rng.gen_bool(0.75) {
                            continue;
                        }

                        let mut graph = org_graph.clone();
                        graph.add_edge(u as Node, u as Node);

                        let graph = $t::from(&graph);
                        assert!(graph.has_node_with_loop());
                        assert_eq!(graph.nodes_with_loops() >> u, 1);
                    }
                }
            }

            #[test]
            fn [< is_chain_node_$n >]() {
                let graph = $t::from(&AdjListMatrix::from(&[(0,1), (1,1), (2,3), (0,3), (3, 4), (4,5), (4,6)]));
                assert!(graph.is_chain_node(0));
                assert!(!graph.is_chain_node(1)); // loop
                assert!(graph.is_chain_node(3));  // two in, one out
                assert!(graph.is_chain_node(4));  // one in, two out
                assert!(graph.is_chain_node(5));

                let graph = $t::from(&AdjListMatrix::from(&[(0,1), (1,1), (2,3), (2,4), (0,3), (3, 4), (3,5), (4,5), (4,6)]));
                assert!(graph.is_chain_node(0));
                assert!(!graph.is_chain_node(1)); // loop
                assert!(!graph.is_chain_node(3)); // two in, two out
                assert!(!graph.is_chain_node(4)); // one in, two out
                assert!(graph.is_chain_node(5));
            }

            #[test]
            fn [<contract_chaining_nodes_$n >]() {
                use rand::prelude::SliceRandom;
                let rng = &mut rand::thread_rng();
                let n = $t::CAPACITY;

                // contract path
                for _ in 0..10 {
                    let random_path = {
                        let mut graph = AdjListMatrix::new(n);
                        let mut nodes = graph.vertices().collect_vec();
                        nodes.shuffle(rng);
                        graph.connect_path(nodes);
                        graph
                    };

                    let graph = $t::from(&random_path);
                    assert_eq!(graph.edges().len(), n - 1);
                    let contracted = graph.contract_chaining_nodes();
                    assert_eq!(contracted.edges().len(), 0);
                }

                // contract cycle
                for _ in 0..10 {
                    let random_path = {
                        let mut graph = AdjListMatrix::new(n);
                        let mut nodes = graph.vertices().collect_vec();
                        nodes.shuffle(rng);
                        graph.connect_cycle(nodes);
                        graph
                    };

                    let graph = $t::from(&random_path);
                    assert_eq!(graph.edges().len(), n);
                    let contracted = graph.contract_chaining_nodes();

                    // we expect exactly one loop to remain
                    let edges = contracted.edges();
                    assert_eq!(edges.len(), 1);
                    assert_eq!(edges[0].0, edges[0].1);
                }
            }

            #[test]
            fn [<node_with_most_undirected_edges $n >]() {
                use rand::Rng;
                let rng = &mut rand::thread_rng();

                for _ in 0..100 {
                    let n = rng.gen_range(1..$t::CAPACITY);
                    let org_graph : AdjListMatrix = crate::random_models::gnp::generate_gnp(rng, $t::CAPACITY as Node, 0.5 / (n as f64));
                    let undirected_degs = org_graph.vertices().map(|u| {
                        org_graph.out_neighbors(u).filter(|&v| org_graph.has_edge(v, u)).count() as Node
                    }).collect_vec();

                    let max_node = $t::from(&org_graph).node_with_most_undirected_edges();

                    let max_deg = *undirected_degs.iter().max().unwrap();
                    if max_deg == 0 {
                        assert!(max_node.is_none());
                    } else {
                        assert_eq!(undirected_degs[max_node.unwrap() as usize], max_deg);
                    }
                }
            }

            #[test]
            fn [<swap_nodes_ $n >]() {
                use rand::Rng;
                let rng = &mut rand::thread_rng();

                for _ in 0..100 {
                    let n = rng.gen_range(2..$t::CAPACITY);
                    let org_graph : AdjListMatrix = crate::random_models::gnp::generate_gnp(rng, $t::CAPACITY as Node, 0.5 / (n as f64));

                    let u = rng.gen_range(0..n) as Node;
                    let v = loop {let v = rng.gen_range(0..n) as Node; if u != v {break v}};

                    let mut ranks = org_graph.vertices().collect_vec();
                    ranks[u as usize] = v;
                    ranks[v as usize] = u;

                    let mapper = NodeMapper::from_rank(&ranks);
                    let relabelled : AdjListMatrix = mapper.relabelled_graph_as(&org_graph);

                    //

                    let graph = $t::from(&org_graph).swap_nodes(u, v);
                    assert_eq!(graph.edges(), relabelled.edges_vec());
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
