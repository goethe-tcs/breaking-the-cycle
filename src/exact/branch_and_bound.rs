use crate::graph::*;
use crate::utils::int_iterator::IntegerIterators;
use bitintr::{Pdep, Pext};
use num::cast::AsPrimitive;
use num::{FromPrimitive, Integer, PrimInt};
use std::ops::{BitOrAssign, Range, ShlAssign};

/// Return the smallest dfvs with up to `upper_bound` nodes (inclusive).
pub fn branch_and_bound<G: AdjacencyList>(
    graph: &G,
    upper_bound: Option<Node>,
) -> Option<Vec<Node>> {
    let upper_bound = upper_bound.unwrap_or(graph.len() as Node);

    let solution = if graph.len() > 32 {
        let graph: GenericIntGraph<u64, 64> = GenericIntGraph::from(graph);
        branch_and_bound_impl_sccs(&graph, upper_bound)
    } else if graph.len() > 16 {
        let graph: GenericIntGraph<u32, 32> = GenericIntGraph::from(graph);
        branch_and_bound_impl_sccs(&graph, upper_bound)
    } else if graph.len() > 8 {
        let graph: GenericIntGraph<u16, 16> = GenericIntGraph::from(graph);
        branch_and_bound_impl_sccs(&graph, upper_bound)
    } else {
        let graph: GenericIntGraph<u8, 8> = GenericIntGraph::from(graph);
        branch_and_bound_impl_sccs(&graph, upper_bound)
    }?;

    Some(solution.included())
}

trait GraphItem:
    PrimInt
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

trait BBSolver {
    fn branch_and_bound(graph: &Self, limit: Node) -> Solution;
}

trait BBGraph<T: GraphItem, const N: usize> {
    type SccIterator: Iterator<Item = T>;

    fn len(&self) -> usize;
    fn remove_first_node(&self) -> Self;
    fn contract_first_node(&self) -> Self;
    fn nodes_on_cycle(&self, transitive_closure: Option<[T; N]>) -> T;
    fn vertices(&self) -> Range<usize> {
        0..self.len()
    }
    fn transitive_closure(&self) -> [T; N];
    fn sccs(&self, transitive_closure: Option<[T; N]>) -> Self::SccIterator;
    fn subgraph(&self, included_nodes: T) -> Self;
    fn remove_loops(&self) -> (T, T, Self);
    fn first_node_has_loop(&self) -> bool;
    fn nodes_mask(&self) -> T;
    fn has_any_nodes_on_cycle(&self, transitive_closure: Option<[T; N]>) -> bool;
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct GenericIntGraph<T: GraphItem, const N: usize> {
    matrix: [T; N],
    n: usize,
}

impl<T: GraphItem, const N: usize> GenericIntGraph<T, N> {
    fn as_graph<F, const M: usize>(&self) -> GenericIntGraph<F, M>
    where
        T: GraphItem + AsPrimitive<F>,
        F: GraphItem,
    {
        assert!(M >= self.n);
        let mut matrix = [F::zero(); M];

        matrix
            .iter_mut()
            .zip(self.matrix.iter())
            .take(self.n)
            .for_each(|(new_row, old_row)| *new_row = old_row.as_());

        GenericIntGraph { matrix, n: self.n }
    }
}

trait BBSolution: Sized + Clone {
    fn new() -> Self;
    fn insert_new_node(&mut self, i: Node) -> &mut Self;
    fn insert_new_nodes(&mut self, nodes: u64) -> &mut Self;
    fn cardinality(&self) -> Node;
    fn included(&self) -> Vec<Node>;
    fn merge(&mut self, s: Self) -> &mut Self;
    fn shift_from_subgraph(&mut self, subgraph: u64) -> &mut Self;
}

#[derive(Clone, Debug, PartialEq)]
struct Solution {
    nodes_in_dfvs: u64,
    cardinality: u32,
}

impl BBSolution for Solution {
    fn new() -> Self {
        Self {
            nodes_in_dfvs: 0,
            cardinality: 0,
        }
    }

    /// Add a node to the solution. Precondition: The node is not in the solution yet.
    fn insert_new_node(&mut self, i: Node) -> &mut Self {
        debug_assert!(i < 64);
        debug_assert!(self.nodes_in_dfvs & (1_u64 << i as usize) == 0);
        self.nodes_in_dfvs |= 1_u64 << i as usize;
        self.cardinality += 1;
        self
    }

    /// Add new nodes to the solution. Precondition: None of the nodes are in the solution yet.
    fn insert_new_nodes(&mut self, nodes: u64) -> &mut Self {
        debug_assert!(self.nodes_in_dfvs & nodes as u64 == 0);
        self.nodes_in_dfvs |= nodes as u64;
        self.cardinality = self.nodes_in_dfvs.count_ones();

        self
    }

    fn cardinality(&self) -> Node {
        self.cardinality
    }

    fn included(&self) -> Vec<Node> {
        self.nodes_in_dfvs.iter_ones().collect()
    }

    fn merge(&mut self, s: Self) -> &mut Self {
        let included: u64 = self.nodes_in_dfvs | s.nodes_in_dfvs;

        self.nodes_in_dfvs = included;
        self.cardinality = included.count_ones();

        self
    }

    /// Assume the current solution `self` is a solution to a Graph `H` that is a subgraph of
    /// a graph `G`. Then given a bitvector `subgraph` describing the nodes of `H` in `G`,
    /// the function modifies the solution inplace to contain the nodes in `G`
    /// that correspond to the nodes in `H` that are currently contained.
    ///
    /// For example, if `H` is the subgraph of `G` containing nodes `{0, 3, 5, 7}`,
    /// and the solution currently contains nodes `{1, 2}`, then this modifies the solution
    /// to contain nodes `{3, 5}`, as the nodes `{1, 2}` in the subgraph correspond to nodes
    /// `{3, 5}` in the original graph.
    fn shift_from_subgraph(&mut self, subgraph: u64) -> &mut Self {
        self.nodes_in_dfvs = self.nodes_in_dfvs.pdep(subgraph);
        self
    }
}

impl<I: AdjacencyList, T: GraphItem, const N: usize> From<&I> for GenericIntGraph<T, N> {
    fn from(graph: &I) -> Self {
        debug_assert!(graph.len() <= N);

        let mut matrix = [T::zero(); N];

        for (u, neighbors) in graph.vertices().zip(matrix.iter_mut()) {
            for v in graph.out_neighbors(u) {
                *neighbors |= T::one() << v as usize;
            }
        }

        Self {
            matrix,
            n: graph.len(),
        }
    }
}

struct GenericIntGraphSccIterator<T: GraphItem, const N: usize> {
    original_graph: GenericIntGraph<T, N>,
    transitive_closure: [T; N],
    added_to_an_scc: T,
    original_graph_nodes_mask: T,
}

impl<T: GraphItem, const N: usize> GenericIntGraphSccIterator<T, N> {
    pub fn new(g: &GenericIntGraph<T, N>, transitive_closure: Option<[T; N]>) -> Self {
        Self {
            original_graph: *g,
            transitive_closure: transitive_closure.unwrap_or_else(|| g.transitive_closure()),
            added_to_an_scc: T::zero(),
            original_graph_nodes_mask: g.nodes_mask(),
        }
    }

    /// Returns the SCC containing `start_node`, where `start_node` is the minimal id included
    /// in the SCC.
    fn scc(&mut self, start_node: Node) -> T {
        let start_node = start_node as usize;
        let start_node_paths = self.transitive_closure[start_node];

        let mut this_scc: T = T::one() << start_node;

        // Add all nodes to the scc where paths (start_node, node) and (node, start_node) exist.
        let mut node_mask = T::one() << (start_node + 1);
        for &node_paths in &self.transitive_closure[start_node + 1..self.original_graph.len()] {
            if (self.added_to_an_scc & node_mask) == T::zero()
                && (start_node_paths & node_mask) != T::zero()
                && (node_paths & (T::one() << start_node)) != T::zero()
            {
                this_scc |= node_mask;
            }

            node_mask <<= T::one();
        }

        self.added_to_an_scc |= this_scc;

        this_scc
    }
}

impl<T: GraphItem, const N: usize> Iterator for GenericIntGraphSccIterator<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.added_to_an_scc == self.original_graph_nodes_mask {
            return None;
        }

        let start_node = self.added_to_an_scc.trailing_ones() as Node;
        let next_scc = self.scc(start_node);

        Some(next_scc)
    }
}

impl<T: GraphItem, const N: usize> BBGraph<T, N> for GenericIntGraph<T, N> {
    type SccIterator = GenericIntGraphSccIterator<T, N>;

    fn len(&self) -> usize {
        self.n
    }

    fn remove_first_node(&self) -> Self {
        let mut matrix = [T::zero(); N];
        matrix
            .iter_mut()
            .zip(self.matrix[1..].iter())
            .for_each(|(this_row, next_row)| *this_row = *next_row >> 1);

        Self {
            matrix,
            n: self.n - 1,
        }
    }

    /// Contract the first node and remove any loops that result from the contraction.
    /// Return the absolute deleted nodes due to loops, and the resulting graph.
    fn contract_first_node(&self) -> Self {
        debug_assert!(!self.first_node_has_loop());
        let first_node_mask = self.matrix[0];
        let mut matrix = [T::zero(); N];

        matrix
            .iter_mut()
            .zip(self.matrix[1..].iter())
            .for_each(|(this_row, next_row)| {
                *this_row = (*next_row | ((*next_row & T::one()) * first_node_mask)) >> 1;
            });

        Self {
            matrix,
            n: self.n - 1,
        }
    }

    fn nodes_on_cycle(&self, transitive_closure: Option<[T; N]>) -> T {
        if self.len() == 1 {
            return self.matrix[0] & T::one();
        }

        let transitive_closure = transitive_closure.unwrap_or_else(|| self.transitive_closure());

        self.vertices()
            .map(|i| transitive_closure[i] & (T::one() << i))
            .fold(T::zero(), |a, b| a | b)
    }

    fn transitive_closure(&self) -> [T; N] {
        let mut matrix1 = [T::zero(); N];
        let mut matrix2 = self.matrix;

        let matrix_mul = |mat_in: &[T; N], mat_out: &mut [T; N]| {
            for i in self.vertices() {
                mat_out[i] = mat_in[i];
                for j in self.vertices() {
                    mat_out[i] |= mat_in[j] * ((mat_in[i] >> j) & T::one());
                }
            }
        };

        for i in 0..N {
            if i.is_even() {
                matrix_mul(&matrix2, &mut matrix1);
            } else {
                matrix_mul(&matrix1, &mut matrix2);
            }
        }

        if N.is_odd() {
            matrix1
        } else {
            matrix2
        }
    }

    fn sccs(&self, transitive_closure: Option<[T; N]>) -> Self::SccIterator {
        Self::SccIterator::new(self, transitive_closure)
    }

    fn subgraph(&self, included_nodes: T) -> Self {
        if included_nodes == T::zero() {
            return Self {
                matrix: [T::zero(); N],
                n: 0,
            };
        }

        if included_nodes == self.nodes_mask() {
            return *self;
        }

        let mut matrix = [T::zero(); N];
        let n = included_nodes.count_ones() as usize;

        for (old_row, new_row) in included_nodes.iter_ones().zip(matrix.iter_mut()) {
            *new_row = self.matrix[old_row as usize].pext(included_nodes);
        }

        Self { matrix, n }
    }

    /// Remove loops and return the loops, the mask of the resulting subgraph
    /// without loops, and the resulting subgraph.
    fn remove_loops(&self) -> (T, T, Self) {
        let loops: T = self.matrix[0..self.n]
            .iter()
            .enumerate()
            .map(|(i, row)| *row & (T::one() << i))
            .fold(T::zero(), |a, b| a | b);

        let nodes_without_loops = !loops & self.nodes_mask();

        (
            loops,
            nodes_without_loops,
            self.subgraph(nodes_without_loops),
        )
    }

    fn first_node_has_loop(&self) -> bool {
        (self.matrix[0] & T::one()) != T::zero()
    }

    fn nodes_mask(&self) -> T {
        if self.n == N {
            T::max_value()
        } else {
            (T::one() << self.n) - T::one()
        }
    }

    fn has_any_nodes_on_cycle(&self, transitive_closure: Option<[T; N]>) -> bool {
        self.nodes_on_cycle(transitive_closure) != T::zero()
    }
}

fn branch_and_bound_impl_sccs<T: GraphItem, const N: usize>(
    graph: &GenericIntGraph<T, N>,
    mut limit: Node,
) -> Option<Solution> {
    let mut solution = Solution::new();
    let transitive_closure = graph.transitive_closure();

    if !graph.has_any_nodes_on_cycle(Some(transitive_closure)) {
        return Some(solution);
    }

    if limit == 0 {
        return None;
    }

    for scc in graph.sccs(Some(transitive_closure)) {
        let scc_graph = graph.subgraph(scc);

        let mut scc_solution = branch_and_bound_impl(&scc_graph, limit)?;
        scc_solution.shift_from_subgraph(scc.as_());

        limit -= scc_solution.cardinality();
        solution.merge(scc_solution);
    }

    Some(solution)
}

fn branch_and_bound_impl<T: GraphItem, const N: usize>(
    graph: &GenericIntGraph<T, N>,
    limit: Node,
) -> Option<Solution> {
    if graph.len() <= 8 {
        if N > 8 {
            return branch_and_bound_impl(&graph.as_graph::<u8, 8>(), limit);
        }
    } else if graph.len() <= 16 {
        if N > 16 {
            return branch_and_bound_impl(&graph.as_graph::<u16, 16>(), limit);
        }
    } else if graph.len() <= 32 && N > 32 {
        return branch_and_bound_impl(&graph.as_graph::<u32, 32>(), limit);
    }

    if !graph.has_any_nodes_on_cycle(None) {
        return Some(Solution::new());
    }

    if limit == 0 {
        return None;
    }

    let branch1_closure = |branch1_subgraph, limit, branch1_subgraph_mask: T| {
        let mut solution1 = branch_and_bound_impl_sccs::<T, N>(branch1_subgraph, limit - 1)?;
        solution1
            .shift_from_subgraph(branch1_subgraph_mask.as_())
            .insert_new_node(0);
        Some(solution1)
    };

    let branch2_closure = |branch2_subgraph, limit, branch2_subgraph_mask: T, loops_mask: T| {
        if loops_mask.count_ones() >= limit {
            // This can happen if solution1 is already better than the current solution
            // plus the self-loops from contracting.
            return None;
        }

        let mut solution2 =
            branch_and_bound_impl::<T, N>(branch2_subgraph, limit - loops_mask.count_ones())?;
        solution2.shift_from_subgraph(branch2_subgraph_mask.as_());
        solution2.insert_new_nodes(loops_mask.as_());
        Some(solution2)
    };

    let branch1_graph = graph.remove_first_node();
    let branch1_subgraph_mask = graph.nodes_mask() - T::one();

    let solution1 = branch1_closure(&branch1_graph, limit, branch1_subgraph_mask);
    let limit = solution1.as_ref().map_or(limit, |x| x.cardinality());

    if graph.first_node_has_loop() {
        return solution1;
    }

    let branch2_graph = graph.contract_first_node();
    let (loops_mask, subgraph_without_loops_mask, branch2_graph) = branch2_graph.remove_loops();
    let branch2_subgraph_mask = subgraph_without_loops_mask << 1;
    let branch2_loops_mask = loops_mask << 1;

    let solution2 = branch2_closure(
        &branch2_graph,
        limit,
        branch2_subgraph_mask,
        branch2_loops_mask,
    );

    if solution2.as_ref().map_or(limit, |x| x.cardinality()) < limit {
        solution2
    } else {
        solution1
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bitset::BitSet;
    use crate::graph::generators::GeneratorSubstructures;
    use crate::random_models::gnp::generate_gnp;
    use itertools::Itertools;
    use rand::prelude::SliceRandom;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    impl<T: GraphItem, const N: usize> GenericIntGraph<T, N> {
        fn has_edge(&self, u: usize, v: usize) -> bool {
            (self.matrix[u] >> v) & T::one() == T::one()
        }

        fn edges(&self) -> Vec<Edge> {
            self.vertices()
                .cartesian_product(self.vertices())
                .filter_map(|(i, j)| {
                    if self.has_edge(i, j) {
                        Some((i as Node, j as Node))
                    } else {
                        None
                    }
                })
                .collect()
        }

        fn remove_node(&self, u: Node) -> Self {
            assert!(u < self.n as Node);

            let mut matrix = [T::zero(); N];

            for (i, row) in matrix.iter_mut().enumerate().take(self.n - 1) {
                let o = self.matrix[if i < u as usize { i } else { i + 1 }];
                let mask = (T::one() << u as usize) - T::one();
                // ex.: o = 10110110, u = 2  -> mask = 00000011
                // -> o&mask gives lower bits we want to keep
                // -> (o>>1) & !mask gives higher bits we want to keep, shifted to the right (to skip
                // the node we are deleting)
                *row = (o & mask) | ((o >> 1) & !mask);
            }

            Self {
                matrix,
                n: self.n - 1,
            }
        }
    }

    #[test]
    fn bbgraph_from() {
        let edges = [(0, 1), (1, 1), (2, 3), (6, 5)];
        let graph = GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges));
        assert_eq!(graph.edges(), Vec::from(edges));
    }

    #[test]
    fn bbgraph_acyclic() {
        for i in 1_usize..32 {
            let mut adjmat = AdjListMatrix::new(i);
            let path_vertices: Vec<Node> = adjmat.vertices().collect();
            adjmat.connect_path(path_vertices.into_iter());
            {
                let graph = GenericIntGraph::<u32, 32>::from(&adjmat);
                assert_eq!(graph.nodes_on_cycle(None), 0, "i={}", i);
            }
            adjmat.add_edge(i as Node - 1, 0);
            {
                let graph = GenericIntGraph::<u32, 32>::from(&adjmat);
                assert_eq!(
                    graph.nodes_on_cycle(None),
                    ((1u64 << i) - 1) as u32,
                    "i={}",
                    i
                );
            }
        }
    }

    #[test]
    fn bbgraph_remove_node() {
        let edges = [(0, 1), (1, 1), (2, 3), (6, 5)];
        let graph = GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges));
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
    fn subgraph() {
        let edges = [(0, 1), (0, 4), (1, 1), (2, 3), (6, 5), (6, 7), (7, 5)];
        let graph = GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges));
        let subgraph = graph.subgraph(0);
        assert_eq!(subgraph.n, 0);
        assert_eq!(subgraph.matrix, [0_u32; 32]);

        let subgraph = graph.subgraph(0b00000011);
        assert_eq!(subgraph.n, 2);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[(0, 1), (1, 1)])).matrix
        );

        let subgraph = graph.subgraph(0b00010111);
        assert_eq!(subgraph.n, 4);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[(0, 1), (1, 1), (0, 3)]))
                .matrix
        );

        let subgraph = graph.subgraph(0b11101000);
        assert_eq!(subgraph.n, 4);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[(2, 1), (2, 3), (3, 1)]))
                .matrix
        );

        let subgraph = graph
            .remove_first_node()
            .remove_first_node()
            .subgraph(0b00111111);
        assert_eq!(subgraph.n, 6);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[
                (0, 1),
                (4, 3),
                (4, 5),
                (5, 3)
            ]))
            .matrix
        );

        let subgraph = graph.remove_node(2).remove_node(3).subgraph(0b00111111);
        assert_eq!(subgraph.n, 6);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[
                (0, 1),
                (1, 1),
                (4, 3),
                (4, 5),
                (5, 3)
            ]))
            .matrix
        );

        let subgraph = graph
            .remove_node(0)
            .remove_node(6)
            .remove_node(2)
            .subgraph(0b00011001);
        assert_eq!(subgraph.n, 3);
        assert_eq!(
            subgraph.matrix,
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&[(0, 0), (2, 1)])).matrix
        );
    }

    #[test]
    fn remove_loops() {
        let graph = |edges| GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges));

        let remove_loops_test = |graph: GenericIntGraph<u32, 32>,
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
    fn sccs() {
        let scc_vec = |edges| {
            GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges))
                .sccs(None)
                .map(|scc| {
                    assert!(scc <= 255); // Only graphs with up to 8 nodes to simplify tests
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

    #[test]
    fn shift_from_subgraph() {
        assert_eq!(
            *Solution::new().shift_from_subgraph(0b00000000),
            Solution::new()
        );

        assert_eq!(
            *Solution::new().shift_from_subgraph(0b00101011),
            Solution::new()
        );

        assert_eq!(
            *Solution::new().shift_from_subgraph(u32::MAX as u64),
            Solution::new()
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .shift_from_subgraph(0b00000001),
            *Solution::new().insert_new_node(0)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(0)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(1)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(1)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(2)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(3)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(3)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(5)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(1)
                .insert_new_node(4)
                .insert_new_node(0)
                .shift_from_subgraph(0b10101011),
            *Solution::new()
                .insert_new_node(0)
                .insert_new_node(1)
                .insert_new_node(7)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .insert_new_node(1)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(0).insert_new_node(1)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(3)
                .insert_new_node(7)
                .insert_new_node(1)
                .shift_from_subgraph(0b11111111),
            *Solution::new()
                .insert_new_node(3)
                .insert_new_node(7)
                .insert_new_node(1)
        );
    }

    #[test]
    fn bb() {
        {
            // graph is acyclic -> solution {}
            assert_eq!(
                branch_and_bound(&AdjListMatrix::from(&[(0, 1)]), None).unwrap(),
                vec![]
            );
        }

        {
            // graph has loop at 0 -> solution {0}
            assert_eq!(
                branch_and_bound(&AdjListMatrix::from(&[(0, 1), (0, 0)]), None).unwrap(),
                vec![0]
            );
        }

        {
            // graph has loop at 0 -> solution {0}
            assert_eq!(
                branch_and_bound(&AdjListMatrix::from(&[(0, 1), (0, 0)]), None).unwrap(),
                vec![0]
            );
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            assert_eq!(
                branch_and_bound(&AdjListMatrix::from(&[(0, 1), (0, 0), (3, 3)]), None,).unwrap(),
                vec![0, 3]
            );
        }

        {
            // no solution, as limit too low
            assert!(
                branch_and_bound(&AdjListMatrix::from(&[(0, 1), (0, 0), (3, 3)]), Some(1),)
                    .is_none()
            );
        }

        {
            // graph has loop at 2 -> solution {2}
            let graph = AdjListMatrix::from(&[(0, 1), (1, 2), (2, 3), (3, 0), (2, 2)]);
            assert_eq!(branch_and_bound(&graph, None).unwrap(), vec![2]);
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            let graph = AdjListMatrix::from(&[(0, 0), (1, 2), (2, 3), (3, 4), (4, 1), (3, 3)]);
            assert_eq!(branch_and_bound(&graph, None).unwrap(), vec![0, 3]);
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            let mut nodes = [0, 1, 2, 3, 4, 5];
            let graph = AdjListMatrix::from(&[(0, 0), (1, 2), (2, 3), (3, 4), (4, 1), (5, 5)]);

            for _ in 0..10 {
                nodes.shuffle(&mut rand::thread_rng());
                let solution = branch_and_bound(&graph, None).unwrap();
                assert_eq!(solution.len(), 3);
                assert_eq!(solution[0], 0);
                assert!(1 <= solution[1] && solution[1] < 5);
                assert_eq!(solution[2], 5);
            }
        }
    }

    #[test]
    fn bb_scc_specific() {
        // limit reached in first scc
        assert!(branch_and_bound(
            &AdjListMatrix::from(&[(0, 1), (1, 0), (2, 3), (3, 2)]),
            Some(1),
        )
        .is_none());

        // limit reached in second scc
        assert!(branch_and_bound(
            &AdjListMatrix::from(&[
                (0, 1),
                (1, 0),
                (2, 3),
                (2, 4),
                (3, 3),
                (3, 4),
                (4, 2),
                (5, 5)
            ]),
            Some(3),
        )
        .is_none());

        assert_eq!(
            branch_and_bound(
                &AdjListMatrix::from(&[(0, 1), (0, 2), (1, 1), (1, 2), (2, 0),]),
                Some(3),
            )
            .unwrap(),
            vec![0, 1]
        );

        assert_eq!(
            branch_and_bound(
                &AdjListMatrix::from(&[
                    (0, 1),
                    (1, 0),
                    (2, 3),
                    (2, 4),
                    (3, 3),
                    (3, 4),
                    (4, 2),
                    (5, 5)
                ]),
                Some(4),
            )
            .unwrap(),
            vec![0, 2, 3, 5]
        );

        assert_eq!(
            branch_and_bound(
                &AdjListMatrix::from(&[
                    (0, 3),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (2, 4),
                    (3, 1),
                    (4, 0),
                    (4, 2),
                ]),
                None,
            )
            .unwrap(),
            vec![1, 2]
        );

        assert_eq!(
            branch_and_bound(
                &AdjListMatrix::from(&[(0, 3), (1, 0), (1, 2), (1, 3), (2, 4), (3, 1), (4, 2),]),
                None,
            )
            .unwrap(),
            vec![1, 2]
        );

        // several recursive sccs when removing a node
        assert_eq!(
            branch_and_bound(
                &AdjListMatrix::from(&[
                    (0, 1),
                    (1, 4),
                    (2, 1),
                    (2, 3),
                    (2, 4),
                    (3, 5),
                    (4, 2),
                    (5, 3),
                    (5, 0)
                ]),
                None,
            )
            .unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn bb_generated_tests() {
        // The results were generated by the branch_and_bound implementation in MR19.
        let solution_sizes = vec![
            4, 4, 5, 5, 4, 4, 6, 7, 5, 6, 5, 6, 6, 8, 6, 6, 8, 6, 5, 6, 7, 7, 7, 5, 7, 8, 9, 9, 7,
            7, 9, 9, 8, 10, 8, 8,
        ];

        let mut gen = Pcg64Mcg::seed_from_u64(123);

        for n in 10..=21 {
            for avg_deg in [5] {
                for i in 0..3 {
                    let p = avg_deg as f64 / n as f64;
                    let mut graph: AdjArray = generate_gnp(&mut gen, n, p);

                    for i in graph.vertices_range() {
                        graph.try_remove_edge(i, i);
                    }

                    let solution = branch_and_bound(&graph, None).unwrap();
                    let solution_mask = {
                        let mut set = BitSet::new_all_set(graph.len());
                        for node in &solution {
                            set.unset_bit(*node as usize);
                        }
                        set
                    };

                    assert_eq!(solution.len(), solution_sizes[((n - 10) * 3 + i) as usize]);
                    assert!(graph.vertex_induced(&solution_mask).0.is_acyclic());
                }
            }
        }
    }
}
