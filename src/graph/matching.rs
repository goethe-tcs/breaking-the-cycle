use super::*;
use crate::bitset::BitSet;
use crate::graph::network_flow::{EdmondsKarp, ResidualBitMatrix, ResidualNetwork};
use itertools::Itertools;
use std::iter;

pub trait Matching {
    /// Computes a maximal matching on a graph where all edges are undirected. Each edge {u, v} in
    /// the matching is returned only once as (u, v) where u <= v. The output is sorted lexicographically.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::*;
    /// let graph = AdjArrayUndir::from(&[(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]); // 0 <=> 1 <=> 2 <=> 3
    /// let matching = graph.maximal_undirected_matching();
    /// assert!(matching == vec![(0,1), (2,3)] || matching == vec![(1,2)]);
    /// ```
    fn maximal_undirected_matching(&self) -> Vec<(Node, Node)> {
        self.maximal_undirected_matching_excluding(iter::empty())
    }

    /// Computes a maximal matching on an induced sub graph where all edges are undirected. The induced
    /// subgraph contains all vertices *BUT* the ones provided via the iterator `excl`. Each edge `{u, v}` in
    /// the matching is returned only once as `(u, v)` where `u <= v`. The output is sorted lexicographically.
    /// Observe that the original graph may contain directed edges, only the subgraph must not contain any
    /// edge `(u, v)` without a matching (v, u).
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::*;
    /// let graph = AdjArrayUndir::from(&[(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]); // 0 <=> 1 <=> 2 <=> 3
    /// let matching = graph.maximal_undirected_matching_excluding(std::iter::once(1));
    /// assert_eq!(matching, vec![(2,3)]);
    /// ```
    fn maximal_undirected_matching_excluding<I: Iterator<Item = Node>>(
        &self,
        excl: I,
    ) -> Vec<(Node, Node)>;

    /// Computes a maximum matching on a bipartite (sub)graph. The subgraph consists of two classes
    /// A and B and contains all edges of the original graph pointing from A to B; edges of the original
    /// graph that connect two nodes within the same class are ignored (this includes loops); also edges
    /// pointing from B to A are ignored.
    ///
    /// The maximum matching is returned as tuples of form (node of A, node of B) in no deterministic order.
    /// ```
    /// use dfvs::graph::*;
    /// use dfvs::bitset::BitSet;
    /// let graph = AdjArrayUndir::from(&[(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]); // 0 <=> 1 <=> 2 <=> 3
    /// let matching = graph.maximum_bipartite_matching(
    ///     &BitSet::new_all_unset_but(4, [0usize, 2].iter().copied()),
    ///     &BitSet::new_all_unset_but(4, [1usize, 3].iter().copied())
    /// );
    /// assert_eq!(matching.len(), 2);
    /// ```

    fn maximum_bipartite_matching(&self, class_a: &BitSet, class_b: &BitSet) -> Vec<(Node, Node)>;
}

impl<G> Matching for G
where
    G: AdjacencyList + AdjacencyListIn + AdjacencyListUndir,
{
    fn maximal_undirected_matching_excluding<I: Iterator<Item = Node>>(
        &self,
        excl: I,
    ) -> Vec<(Node, Node)> {
        let mut matching = Vec::new();
        let mut matched = BitSet::new_all_unset_but(self.len(), excl);

        /*
        debug_assert!(
            !self
                .vertices()
                .any(|u| !matched[u as usize] && self.total_degree(u) != 2 * self.undir_degree(u))
        );
         */

        for u in self.vertices() {
            if matched[u as usize] {
                continue;
            }

            if let Some(v) = self.out_neighbors(u).find(|&v| !matched[v as usize]) {
                matched.set_bit(u as usize);
                matched.set_bit(v as usize);
                matching.push((u, v));
            }
        }

        matching
    }

    fn maximum_bipartite_matching(&self, class_a: &BitSet, class_b: &BitSet) -> Vec<(Node, Node)> {
        if class_a.cardinality() == 0 || class_b.cardinality() == 0 {
            return Vec::new();
        }

        debug_assert!(class_a.is_disjoint_with(class_b));

        let n = 2 + class_a.cardinality() + class_b.cardinality();

        // labels
        let labels = [self.number_of_nodes(), self.number_of_nodes() + 1]
            .iter()
            .copied()
            .chain(class_a.iter().chain(class_b.iter()).map(|u| u as Node))
            .collect_vec();

        let capacity = {
            let mut capacity = Vec::with_capacity(n);

            // s is connected to all nodes in class_a; t has only incoming edges
            capacity.push(BitSet::new_all_unset_but(n, 2..2 + class_a.cardinality()));
            capacity.push(BitSet::new(n));

            let mapping_b = {
                let mut mapping_b = vec![n; self.len()];
                for (mapped, org) in class_b.iter().enumerate() {
                    mapping_b[org] = 2 + class_a.cardinality() + mapped;
                }
                mapping_b
            };

            for u in class_a.iter() {
                capacity.push(BitSet::new_all_unset_but(
                    n,
                    self.out_neighbors(u as Node)
                        .map(|v| mapping_b[v as usize])
                        .filter(|&v| v < n),
                ))
            }

            for _ in 0..class_b.cardinality() {
                capacity.push(BitSet::new_all_unset_but(n, iter::once(1usize)))
            }

            capacity
        };

        let network = ResidualBitMatrix::from_capacity_and_labels(capacity, labels, 0, 1);

        let mut ek = EdmondsKarp::new(network);
        for _ in ek.by_ref() {} // execute EK to completion
        let (capacity, labels) = ek.take();

        // iterate over all nodes of class b -- each should have exactly one out neighbor; if its
        // the target t (id = 1), then the node is unmatched; otherwise it's the matching partner
        capacity
            .iter()
            .skip(2 + class_a.cardinality())
            .zip(class_b.iter())
            .filter_map(|(out, node_b)| {
                debug_assert_eq!(out.cardinality(), 1);
                let node_a = out.get_first_set().unwrap();
                if node_a == 1 {
                    None
                } else {
                    Some((labels[node_a], node_b as Node))
                }
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximal_undirected_matching() {
        let graph = AdjArrayUndir::from(&[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximal_undirected_matching();
        assert!(matching == vec![(0, 1), (2, 3)] || matching == vec![(1, 2)]);
    }

    #[test]
    fn maximal_undirected_matching_excluding() {
        let graph = AdjArrayUndir::from(&[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximal_undirected_matching_excluding(std::iter::once(1));
        assert_eq!(matching, vec![(2, 3)]);
    }

    #[test]
    fn maximum_bipartite_matching() {
        let graph = AdjArrayUndir::from(&[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]); // 0 <=> 1 <=> 2 <=> 3
        let matching = graph.maximum_bipartite_matching(
            &BitSet::new_all_unset_but(4, [0usize, 2].iter().copied()),
            &BitSet::new_all_unset_but(4, [1usize, 3].iter().copied()),
        );
        assert_eq!(matching.len(), 2);
    }
}
