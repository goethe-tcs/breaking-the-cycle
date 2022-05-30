use super::*;
use crate::bitset::BitSet;
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

        let mut network = AdjArrayUndir::new(n);
        // edges s -> all nodes in class_a
        for i in 0..class_a.cardinality() {
            network.add_edge(0, 2 + i as Node);
        }

        // edges class_a -> class_b
        {
            let mapping_b = {
                let mut mapping_b = vec![n; self.len()];
                for (mapped, org) in class_b.iter().enumerate() {
                    mapping_b[org] = 2 + class_a.cardinality() + mapped;
                }
                mapping_b
            };

            for (ui, u) in class_a.iter().enumerate() {
                for v in self
                    .out_neighbors(u as Node)
                    .map(|v| mapping_b[v as usize])
                    .filter(|&v| v < n)
                {
                    network.add_edge(2 + ui as Node, v as Node)
                }
            }
        }

        // edges class_b -> t
        for v in 0..class_b.cardinality() {
            network.add_edge((2 + class_a.cardinality() + v) as Node, 1);
        }

        for _ in network.st_flow_keep_changes(|u| labels[u as usize], 0, 1) {} // execute EK to completion

        // iterate over all nodes of class b -- each should have exactly one out neighbor; if its
        // the target t (id = 1), then the node is unmatched; otherwise it's the matching partner
        class_b
            .iter()
            .enumerate()
            .filter_map(|(bi, b)| {
                let bi = (2 + class_a.cardinality() + bi) as Node;
                let b = b as Node;
                debug_assert_eq!(network.out_degree(bi), 1);
                let a = network.out_neighbors(bi).next().unwrap();
                if a == 1 {
                    None
                } else {
                    Some((labels[a as usize], b as Node))
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
