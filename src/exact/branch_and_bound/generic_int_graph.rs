use super::*;

#[derive(Copy, Clone, PartialEq, Debug)]
pub(super) struct GenericIntGraph<T: GraphItem, const N: usize> {
    pub(super) matrix: [T; N],
    n: usize,
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
        let mut matrix = self.matrix;

        // only works for power of two, but integer log is not stable yet
        debug_assert!(N.count_ones() == 1);
        let rounds = N.trailing_zeros();

        for _ in 0..rounds {
            let mut converged = true;
            for i in self.vertices() {
                let org_row = matrix[i]; // //unsafe { *matrix.get_unchecked(i) };
                let new_row = matrix
                    .iter()
                    .enumerate()
                    .map(|(j, &inner_row)| inner_row * ((org_row >> j) & T::one()))
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
            .map(|(i, &row)| row & (T::one() << i))
            .fold(T::zero(), |a, b| a | b)
    }

    fn has_node_with_loop(&self) -> bool {
        self.matrix
            .iter()
            .enumerate()
            .any(|(i, &row)| row & (T::one() << i) != T::zero())
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
            *new_row = self.matrix[old_row as usize].pext(included_nodes);
        }

        Self { matrix, n }
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

    fn out_neighbors(&self, u: Node) -> T {
        self.matrix[u as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;
    use itertools::Itertools;

    type GenericIntGraph8 = GenericIntGraph<u8, 8>;
    type GenericIntGraph16 = GenericIntGraph<u16, 16>;
    type GenericIntGraph32 = GenericIntGraph<u32, 32>;
    type GenericIntGraph64 = GenericIntGraph<u64, 64>;

    super::super::bb_graph::bbgraph_tests!(GenericIntGraph8, u8);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph16, u16);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph32, u32);
    super::super::bb_graph::bbgraph_tests!(GenericIntGraph64, u64);
}
