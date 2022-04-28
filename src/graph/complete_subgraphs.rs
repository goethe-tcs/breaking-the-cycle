use super::*;
use crate::bitset::BitSet;
use itertools::Itertools;
use std::time::{Duration, Instant};

/// Allows to enumerate "directed cliques", i.e. complete induced subgraphs (where we do not
/// require loops).
pub trait CompleteSubgraphEnumerator:
    AdjacencyListIn + AdjacencyListUndir + AdjacencyTest + AdjacencyTestUndir + Sized
{
    /// This function returns an iterator enumerating all maximal directed cliques
    fn enumerate_complete_subgraphs(&self, min_size: Node) -> CompleteSubgraphIterator<Self>;

    /// This function returns a maximum clique (i.e. a largest maximal clique) with a few
    /// optimizations shortcutting the underlying enumeration
    fn maximum_complete_subgraph(&self, max_size: Option<Node>) -> Option<Vec<Node>> {
        self.maximum_complete_subgraph_with_timeout(max_size, None)
            .0
    }

    /// Analogously to [`CompleteSubgraphEnumerator::maximum_complete_subgraph`] with an additional
    /// timeout that breaks of the computation and returns the current best result. The second
    /// entry of the return type is true iff a timeout occurred.
    fn maximum_complete_subgraph_with_timeout(
        &self,
        max_size: Option<Node>,
        timeout: Option<Duration>,
    ) -> (Option<Vec<Node>>, bool) {
        let mut current_best: Option<Vec<Node>> = None;

        let mut iter = self.enumerate_complete_subgraphs(2);
        iter.check_maximality = false;

        if let Some(timeout) = timeout {
            iter.stopping_time = Some(Instant::now() + timeout);
        }

        while let Some(better) = iter.next() {
            iter.increase_min_size_to((better.len() + 1) as Node);
            current_best = Some(better);

            if let Some(max_size) = max_size {
                if current_best.as_ref().unwrap().len() >= max_size as usize {
                    break;
                }
            }
        }

        (current_best, false)
    }

    /// Returns true if all nodes provided form a di-clique, i.e. there exists an edge for each `u`,
    /// `v` in `nodes` with `u != v`. Self loops are ignored if present.
    fn is_complete_subgraph<I: IntoIterator<Item = Node>>(&self, nodes: I) -> bool {
        let nodes = BitSet::new_all_unset_but(self.len(), nodes);

        if nodes.cardinality() < 2 {
            return true;
        }

        if nodes.cardinality() == 2 {
            let u = nodes.get_first_set().unwrap();
            let v = nodes.get_next_set(u + 1).unwrap();
            return self.has_undir_edge(u as Node, v as Node);
        }

        nodes
            .iter()
            .all(|u| self.undir_degree(u as Node) + 1 >= nodes.cardinality() as Node)
            && nodes.iter().all(|u| {
                self.undir_neighbors(u as Node)
                    .filter(|&v| nodes[v as usize])
                    .count()
                    + 1
                    >= nodes.cardinality() // if there's a self loop we count it's + 1
            })
    }

    /// Computes the h-index of the degree distribution, i.e. the number `H` such that there exist
    /// `H` different nodes with degree at least `H`. Thus `H+1` is an upper bound on the maximal
    /// clique size.
    fn h_index(&self) -> Node {
        let mut degrees = self
            .vertices()
            .map(|u| self.undir_degree(u))
            .filter(|&d| d > 0)
            .collect_vec();

        if degrees.is_empty() {
            return 0;
        }

        degrees.sort_unstable_by(|a, b| b.cmp(a)); // reverse sort
        degrees
            .iter()
            .enumerate()
            .position(|(i, &d)| i >= d as usize)
            .unwrap() as Node
    }
}

impl<G: AdjacencyListIn + AdjacencyListUndir + AdjacencyTest + AdjacencyTestUndir + Sized>
    CompleteSubgraphEnumerator for G
{
    fn enumerate_complete_subgraphs(&self, min_size: Node) -> CompleteSubgraphIterator<G> {
        CompleteSubgraphIterator::new(self, min_size)
    }
}

pub struct CompleteSubgraphIterator<
    'a,
    G: AdjacencyListIn + AdjacencyListUndir + AdjacencyTestUndir,
> {
    graph: &'a G,
    min_size: Node,
    subgraph: Vec<Node>,
    stack: Vec<(Node, Node)>,
    check_maximality: bool,
    stopping_time: Option<Instant>,
}

impl<'a, G: AdjacencyListIn + AdjacencyListUndir + AdjacencyTest + AdjacencyTestUndir>
    CompleteSubgraphIterator<'a, G>
{
    pub fn new(graph: &'a G, min_size: Node) -> Self {
        let mut result = Self {
            graph,
            min_size,
            stack: vec![],
            subgraph: Vec::with_capacity(min_size as usize),
            check_maximality: true,
            stopping_time: None,
        };

        result.stack = graph
            .vertices_range()
            .filter(|&x| result.has_node_sufficient_degree(x))
            .map(|x| (0, x))
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

    fn is_time_to_stop(&self) -> bool {
        self.stopping_time.map_or(false, |s| s < Instant::now())
    }
}

impl<'a, G: AdjacencyListIn + AdjacencyListUndir + AdjacencyTest + AdjacencyTestUndir> Iterator
    for CompleteSubgraphIterator<'a, G>
{
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // tail recursion is not guaranteed in rust, and so we may not use long recursions
            // to filter out invalid results; so we need to use this loop
            let (base_len, u) = self.stack.pop()?;
            debug_assert!(self.subgraph.len() >= base_len as usize);
            if !self.has_node_sufficient_degree(u) {
                // check is repeated as min_size may have increased
                continue;
            }
            self.subgraph.truncate(base_len as usize);
            self.subgraph.push(u);
            debug_assert!(self.subgraph.iter().is_sorted());

            let stack_size = self.stack.len();

            if self.is_time_to_stop() {
                return None;
            }

            for x in self.graph.undir_neighbors(u) {
                if x <= u
                    || self.graph.undir_degree(x) <= base_len
                    || self
                        .graph
                        .undir_neighbors(x)
                        .filter(|y| *y <= u && self.subgraph.as_slice().binary_search(y).is_ok())
                        .count()
                        < self.subgraph.len()
                {
                    continue;
                }

                self.stack.push((base_len + 1, x));
            }

            if self.stack.len() == stack_size // i.e. no new candidate
                && self.subgraph.len() >= self.min_size as usize
                && (!self.check_maximality || self.is_maximal(&self.subgraph))
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

    fn build_cliques(cliques: &Vec<Vec<Node>>, with_loops: bool) -> AdjArrayUndir {
        let n = *cliques.iter().flatten().max().unwrap() as usize + 1usize;
        let mut graph = AdjArrayUndir::new(n);
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

    #[test]
    fn is_complete_subgraph() {
        let cliques = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8]];
        for with_loops in [false, true] {
            let graph = build_cliques(&cliques, with_loops);
            assert!(graph.is_complete_subgraph(cliques[0].iter().copied()));
            assert!(graph.is_complete_subgraph(cliques[1].iter().copied()));
            assert!(graph.is_complete_subgraph(cliques[2].iter().copied()));
            assert!(graph.is_complete_subgraph(std::iter::empty()));
            assert!(graph.is_complete_subgraph(std::iter::once(1)));

            assert!(!graph.is_complete_subgraph([1, 2]));
            assert!(!graph.is_complete_subgraph([0, 1, 2]));
            assert!(!graph.is_complete_subgraph([0, 2]));
            assert!(!graph.is_complete_subgraph([3, 4, 5, 6, 7, 8]));
        }
    }

    #[test]
    fn h_index() {
        let cliques = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6, 7, 8]];
        let mut graph = build_cliques(&cliques, true);
        assert_eq!(graph.h_index(), 4);
        graph.remove_edges_at_node(8);
        assert_eq!(graph.h_index(), 3);
    }
}
