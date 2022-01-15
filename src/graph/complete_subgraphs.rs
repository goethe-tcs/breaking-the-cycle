use super::*;
use itertools::enumerate;

/// Allows to enumerate "directed cliques", i.e. complete induced subgraphs (where we do not
/// require loops).
pub trait CompleteSubgraphEnumerator: AdjacencyList + AdjacencyTest + Sized {
    /// This function returns an iterator enumerating all maximal directed cliques
    fn enumerate_complete_subgraphs(&self, min_size: Node) -> CompleteSubgraphIterator<Self>;

    /// This function returns a maximum clique (i.e. a largest maximal clique) with a few
    /// optimizations shortcutting the underlying enumeration
    fn maximum_complete_subgraph(&self) -> Option<Vec<Node>> {
        let mut current_best: Option<Vec<Node>> = None;

        let mut iter = self.enumerate_complete_subgraphs(2);
        while let Some(better) = iter.next() {
            iter.increase_min_size_to((better.len() + 1) as Node);
            current_best = Some(better);
        }

        current_best
    }
}

impl<G: AdjacencyList + AdjacencyTest + Sized> CompleteSubgraphEnumerator for G {
    fn enumerate_complete_subgraphs(&self, min_size: Node) -> CompleteSubgraphIterator<G> {
        CompleteSubgraphIterator::new(self, min_size)
    }
}

pub struct CompleteSubgraphIterator<'a, G: AdjacencyList> {
    graph: &'a G,
    min_size: Node,
    subgraph: Vec<Node>,
    stack: Vec<(Node, Node)>,
}

impl<'a, G: AdjacencyList + AdjacencyTest> CompleteSubgraphIterator<'a, G> {
    pub fn new(graph: &'a G, min_size: Node) -> Self {
        let mut result = Self {
            graph,
            min_size,
            stack: vec![],
            subgraph: Vec::with_capacity(min_size as usize),
        };

        result.stack = graph
            .vertices_range()
            .filter(|&x| result.has_node_sufficient_degree(x))
            .map(|x| (0 as Node, x))
            .collect();

        result
    }

    fn increase_min_size_to(&mut self, min_size: Node) {
        assert!(min_size >= self.min_size);
        self.min_size = min_size;
    }

    fn is_maximal(&self, nodes: &[Node]) -> bool {
        if nodes.is_empty() {
            return true;
        }

        let &u = nodes
            .iter()
            .min_by_key(|&&u| self.graph.out_degree(u))
            .unwrap();

        !self
            .graph
            .out_neighbors(u)
            .filter(|&x| !nodes.iter().any(|&y| x == y))
            .any(|x| {
                nodes
                    .iter()
                    .all(|&y| self.graph.has_edge(x, y) && self.graph.has_edge(y, x))
            })
    }

    fn has_node_sufficient_degree(&self, u: Node) -> bool {
        self.graph.out_degree(u) + (!self.graph.has_edge(u, u) as Node) >= self.min_size
    }
}

impl<'a, G: AdjacencyList + AdjacencyTest> Iterator for CompleteSubgraphIterator<'a, G> {
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // tail recursion is not guaranteed in rust, and so we may not use long recursions
            // to filter out invalid results; so we need to use this loop
            let (base_len, u) = self.stack.pop()?;
            assert!(self.subgraph.len() >= base_len as usize);
            if !self.has_node_sufficient_degree(u) {
                // check is repeated as min_size may have increased
                continue;
            }
            self.subgraph.truncate(base_len as usize);
            self.subgraph.push(u);

            let mut found_candidates = false;
            for x in self.graph.out_neighbors(u) {
                if x <= u
                    || self.graph.out_degree(x) <= base_len
                    || !self.subgraph.iter().all(|&y| self.graph.has_edge(x, y))
                {
                    continue;
                }

                self.stack.push((base_len + 1, x));
                found_candidates = true;
            }

            if !found_candidates
                && self.subgraph.len() >= self.min_size as usize
                && self.is_maximal(&self.subgraph)
            {
                return Some(self.subgraph.clone());
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::bitset::BitSet;
    use crate::graph::generators::GeneratorSubstructures;

    fn build_cliques(cliques: &Vec<Vec<Node>>, with_loops: bool) -> AdjListMatrix {
        let n = *cliques.iter().flatten().max().unwrap() as usize + 1usize;
        let mut graph = AdjListMatrix::new(n);
        for c in cliques {
            graph.connect_nodes(&BitSet::new_all_unset_but(n, c.clone()), with_loops);
        }
        graph
    }

    #[test]
    fn disjoints_subgraph() {
        let cliques = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8]];
        for with_loops in [false, true] {
            let graph = build_cliques(&cliques, with_loops);
            let mut found: Vec<Vec<Node>> = CompleteSubgraphIterator::new(&graph, 2).collect();
            found.sort();
            assert_eq!(cliques, found);
        }
    }

    #[test]
    fn overlapping_subgraph() {
        let cliques = vec![vec![0, 1], vec![0, 2, 3], vec![1, 4, 5], vec![2, 3, 4, 5]];
        for with_loops in [false, true] {
            let graph = build_cliques(&cliques, with_loops);
            let mut found: Vec<Vec<Node>> = graph.enumerate_complete_subgraphs(2).collect();
            found.sort();
            assert_eq!(cliques, found);
        }
    }
}
