use super::*;

#[derive(Copy, Clone, PartialEq, Debug, Eq)]
#[repr(C)]
#[repr(align(32))] // align the matrix in a way that we can use efficient SIMD access instructions
pub struct GenericIntGraph<T: GraphItem, const N: usize> {
    pub matrix: [T; N],
    n: usize,
}

impl<I: AdjacencyList, T: GraphItem, const N: usize> From<&I> for GenericIntGraph<T, N> {
    fn from(graph: &I) -> Self {
        debug_assert!(graph.len() <= N);

        let mut matrix = [T::zero(); N];

        for (u, neighbors) in graph.vertices().zip(matrix.iter_mut()) {
            for v in graph.out_neighbors(u) {
                *neighbors |= T::ith_bit_set(v as usize);
            }
        }

        Self {
            matrix,
            n: graph.len(),
        }
    }
}

macro_rules! impl_avx2_dispatch {
    ($self:ident, $n:expr) => {
        paste::item! {
            if Self::CAPACITY == $n {
                let mut tc = *$self; // we need a Graph object to have proper alignment for the matrix
                unsafe {
                     super::avx2::[< transitive_closure$n >] (
                        tc.matrix.as_mut_ptr() as *mut [< u$n >],
                        $self.n as Node,
                    )
                };
                return tc;
            }
        }
    };
}

impl<T: GraphItem, const N: usize> GenericIntGraph<T, N> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            matrix: [T::zero(); N],
        }
    }

    pub fn add_edge(&mut self, u: Node, v: Node) {
        self.matrix[u as usize] |= T::ith_bit_set(v as usize);
    }
}

impl<T: GraphItem, const N: usize> BBGraph for GenericIntGraph<T, N> {
    type NodeMask = T;
    type SccIterator<'a> = SCCIterator<'a, Self>;
    const CAPACITY: usize = N;

    fn from_bbgraph<G: BBGraph>(graph: &G) -> Self
    where
        <G as BBGraph>::NodeMask: num::traits::AsPrimitive<Self::NodeMask>,
    {
        debug_assert!(graph.len() <= N);

        let mut matrix = [T::zero(); N];

        for (row_in, row_out) in graph
            .vertices()
            .map(|u| graph.out_neighbors(u as Node))
            .zip(matrix.iter_mut())
            .take(N)
        {
            *row_out = row_in.as_();
        }

        GenericIntGraph {
            matrix,
            n: graph.len(),
        }
    }

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

    fn remove_node(&self, u: Node) -> Self {
        assert!(u < self.n as Node);

        let mut matrix = [T::zero(); N];

        for (i, row) in matrix.iter_mut().enumerate().take(self.n - 1) {
            let o = self.matrix[if i < u as usize { i } else { i + 1 }];
            let mask = T::ith_bit_set(u as usize) - T::one();
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

    fn transitive_closure(&self) -> Self {
        // first try to dispatch to the AVX2 implementation; observe that the linear search down
        // below should be removed completely by the compiler
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            impl_avx2_dispatch!(self, 16);
            impl_avx2_dispatch!(self, 32);
            impl_avx2_dispatch!(self, 64);
            impl_avx2_dispatch!(self, 128);
        }

        let mut matrix = self.matrix;

        // only works for power of two, but integer log is not stable yet
        debug_assert!(N.count_ones() == 1);
        let rounds = N.trailing_zeros();

        let full_nodes = self.nodes_mask();

        for _ in 0..rounds {
            let mut converged = true;
            for i in self.vertices() {
                let org_row = matrix[i];

                if org_row == full_nodes {
                    continue;
                }

                let new_row = matrix
                    .iter()
                    .take(self.n as usize)
                    .enumerate()
                    .map(|(j, &inner_row)| inner_row * org_row.ith_bit(j))
                    .fold(org_row, |a, b| a | b);

                matrix[i] = new_row;
                converged &= new_row == org_row;
            }

            if converged {
                break;
            }
        }

        Self { matrix, n: self.n }
    }

    fn nodes_with_loops(&self) -> T {
        self.matrix
            .iter()
            .enumerate()
            .map(|(i, &row)| row & T::ith_bit_set(i))
            .fold(T::zero(), |a, b| a | b)
    }

    fn has_node_with_loop(&self) -> bool {
        self.matrix
            .iter()
            .enumerate()
            .any(|(i, &row)| row.is_ith_bit_set(i))
    }

    fn has_all_edges(&self) -> bool {
        let mask = self.nodes_mask();
        self.matrix.iter().all(|x| *x == mask)
    }

    fn sccs(&self) -> Self::SccIterator<'_> {
        Self::SccIterator::new(self)
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
            *new_row = self.matrix[old_row as usize];
        }

        multi_bit_extract(included_nodes, &mut matrix[0..n]);

        Self { matrix, n }
    }

    fn first_node_has_loop(&self) -> bool {
        self.matrix[0].is_ith_bit_set(0)
    }

    fn nodes_mask(&self) -> T {
        T::lowest_bits_set(self.n)
    }

    fn out_neighbors(&self, u: Node) -> T {
        self.matrix[u as usize]
    }

    fn remove_edges_at_nodes(&self, delete: Self::NodeMask) -> Self {
        if delete.is_zero() {
            return *self;
        }

        let mut matrix = self.matrix;
        for x in &mut matrix[0..self.n] {
            *x = *x & !delete;
        }

        for i in delete.iter_ones() {
            matrix[i as usize].set_zero();
        }

        Self { matrix, n: self.n }
    }

    fn contract_chaining_nodes(&self) -> Self {
        // this implementation is not exhaustive in the sense that once a in/out-degree=0 node is
        // found, other nodes may qualify; but such chaining is ignored. Benchmarks suggests it's
        // overall faster to deal with them in the next iteration of the b&b algorithm
        let mut matrix = self.matrix;

        // start with out-degree = 1 nodes
        for i in self.vertices() {
            let out = matrix[i];
            let node = Self::NodeMask::ith_bit_set(i);

            if out != node && out.count_ones() <= 1 {
                // we have exact one out-neighbor which is not a self-loop
                for m in matrix.iter_mut().take(self.n) {
                    *m = (*m & !node) | (out * m.ith_bit(i));
                }

                matrix[i].set_zero();
            }
        }

        let mut has_loop = T::zero();
        'outer: loop {
            // we compute the nodes with in-degree exactly one as follows: for each node we have
            // two bits used to count to two. Initially `first` and `second` are all false. The first
            // time we see a node, its bit in `first` goes high. The second time we see it, also its
            // bit in `second` goes high. Hence, the nodes with in-deg 1 are those which have
            // `first` high and `second` low.
            let mut candidates = {
                let mut first = T::zero();
                let mut second = T::zero();

                for &m in &matrix[0..self.n] {
                    second |= first & m;
                    first |= m;
                }

                first & !second
            };

            loop {
                candidates = candidates & !has_loop;

                if candidates.is_zero() {
                    break 'outer;
                }

                let i = candidates.trailing_zeros() as usize;
                let node = T::ith_bit_set(i);
                let out = matrix[i];

                if !((node & out).is_zero()) {
                    // self loop
                    has_loop |= node;
                    continue;
                }

                if let Some(pred) = matrix.iter_mut().find(|x| !((**x & node).is_zero())) {
                    *pred = (*pred ^ node) | out;
                }
                matrix[i].set_zero();
                break;
            }
        }

        Self { matrix, n: self.n }
    }

    fn node_with_most_undirected_edges(&self) -> Option<Node> {
        if self.n == 0 {
            return None;
        }

        let mut best_known_deg = 0;

        self.matrix
            .iter()
            .enumerate()
            .filter_map(|(u, &out_neighbors)| {
                let u = u as Node;

                // a wee short-cut: if there are fewer out-neighbors than our current best-known,
                // we do not have to bother computing how many of them are undirected
                if (out_neighbors.count_ones() as Node) < best_known_deg {
                    None
                } else {
                    let undirected_deg = out_neighbors
                        .iter_ones()
                        .filter(|&v| self.has_edge(v as Node, u))
                        .count() as Node;

                    if undirected_deg > best_known_deg {
                        best_known_deg = undirected_deg;
                        Some(u)
                    } else {
                        None
                    }
                }
            })
            .last()
    }

    fn swap_nodes(&self, n0: Node, n1: Node) -> Self {
        if n0 == n1 {
            return *self;
        }

        let n0 = n0 as usize;
        let n1 = n1 as usize;

        let mut matrix = self.matrix;
        for m in &mut matrix {
            *m = m.exchange_bits(n0, n1);
        }

        matrix.swap(n0, n1);

        Self { matrix, n: self.n }
    }
}

pub type Graph8 = GenericIntGraph<u8, 8>;
pub type Graph16 = GenericIntGraph<u16, 16>;
pub type Graph32 = GenericIntGraph<u32, 32>;
pub type Graph64 = GenericIntGraph<u64, 64>;
pub type Graph128 = GenericIntGraph<u128, 128>;

macro_rules! impl_try_compact {
    ($i:ty, $o:ty) => {
        impl BBTryCompact<$o> for $i {
            fn try_compact<I: FnMut(&$o) -> Option<<$o as BBGraph>::NodeMask>>(
                &self,
                mut callback: I,
            ) -> Option<Option<Self::NodeMask>> {
                if Self::CAPACITY <= <$o>::CAPACITY || self.len() > <$o>::CAPACITY {
                    return None;
                }

                Some(callback(&<$o>::from_bbgraph(self)).map(|x| x as Self::NodeMask))
            }
        }
    };
}

impl_try_compact!(Graph128, Graph64);
impl_try_compact!(Graph128, Graph32);
impl_try_compact!(Graph128, Graph16);
impl_try_compact!(Graph128, Graph8);

impl_try_compact!(Graph64, Graph64);
impl_try_compact!(Graph64, Graph32);
impl_try_compact!(Graph64, Graph16);
impl_try_compact!(Graph64, Graph8);

impl_try_compact!(Graph32, Graph64);
impl_try_compact!(Graph32, Graph32);
impl_try_compact!(Graph32, Graph16);
impl_try_compact!(Graph32, Graph8);

impl_try_compact!(Graph16, Graph64);
impl_try_compact!(Graph16, Graph32);
impl_try_compact!(Graph16, Graph16);
impl_try_compact!(Graph16, Graph8);

impl_try_compact!(Graph8, Graph64);
impl_try_compact!(Graph8, Graph32);
impl_try_compact!(Graph8, Graph16);
impl_try_compact!(Graph8, Graph8);

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;
    use itertools::Itertools;

    type GenericIntGraph8 = GenericIntGraph<u8, 8>;
    type GenericIntGraph16 = GenericIntGraph<u16, 16>;
    type GenericIntGraph32 = GenericIntGraph<u32, 32>;
    type GenericIntGraph64 = GenericIntGraph<u64, 64>;
    type GenericIntGraph128 = GenericIntGraph<u128, 128>;

    super::super::bb_graph::bbgraph_tests!(GenericIntGraph8, u8);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph16, u16);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph32, u32);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph64, u64);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph128, u128);
}
