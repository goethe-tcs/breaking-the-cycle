use crate::graph::{AdjacencyListIn, GraphEdgeEditing, Node, Traversal};
use fxhash::{FxBuildHasher, FxHashSet};
use keyed_priority_queue::KeyedPriorityQueue;
use std::iter::FromIterator;
use std::marker::PhantomData;

/// Encapsulates the functionality of assessing the value of a vertex when
/// deciding about what vertex to add to a heuristic DFVS solution.
pub trait Selector<G: AdjacencyListIn + GraphEdgeEditing> {
    /// Creates a new Selector instance. Takes ownership of a 'graph-like' structure.
    fn new(graph: G) -> Self;

    /// Returns the best vertex to be added to a dfvs according to some heuristic.
    fn best(&self) -> Node;

    /// Removes the best vertex according to the implemented internal heuristic
    fn remove_best(&mut self) -> Node {
        let u = self.best();
        self.graph_mut().remove_edges_at_node(u);
        u
    }

    /// Returns a reference to the current graph
    fn graph(&self) -> &G;

    /// Returns a mutable reference to the current graph
    fn graph_mut(&mut self) -> &mut G;
}

struct SelectorIterator<G: AdjacencyListIn + GraphEdgeEditing, S: Selector<G>> {
    phantom: PhantomData<G>,
    selector: S,
}

impl<G: AdjacencyListIn + GraphEdgeEditing, S: Selector<G>> SelectorIterator<G, S> {
    fn new(selector: S) -> Self {
        Self {
            phantom: Default::default(),
            selector,
        }
    }
}

impl<G: AdjacencyListIn + GraphEdgeEditing, S: Selector<G>> Iterator for SelectorIterator<G, S> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        if self.selector.graph().is_acyclic() {
            None
        } else {
            Some(self.selector.remove_best())
        }
    }
}

/// Selects the vertex with the largest total degree
/// Internally keeps a queue consisting of the total_degree of every vertex.
/// Updates only the values of the neighbors of a vertex v
pub struct MaxDegreeSelector<G: AdjacencyListIn + GraphEdgeEditing> {
    graph: G,
    queue: KeyedPriorityQueue<Node, Node, FxBuildHasher>,
}

impl<G: AdjacencyListIn + GraphEdgeEditing> Selector<G> for MaxDegreeSelector<G> {
    fn new(graph: G) -> Self {
        let n = graph.len();
        let mut queue = KeyedPriorityQueue::with_capacity_and_hasher(n, FxBuildHasher::default());
        for v in graph.vertices() {
            queue.push(v, graph.total_degree(v));
        }
        Self { graph, queue }
    }

    fn graph(&self) -> &G {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    fn remove_best(&mut self) -> Node {
        let (u, _) = self
            .queue
            .pop()
            .expect("Expected to queue to be non-empty!");
        // TODO: Check with benchmarks if this is slower than simply updating the priority of elements twice
        let nb: FxHashSet<_> = self
            .graph
            .in_neighbors(u)
            .chain(self.graph.out_neighbors(u))
            .collect();
        self.graph.remove_edges_at_node(u);
        for v in nb {
            self.queue
                .set_priority(&v, self.graph.total_degree(v))
                .unwrap_or_else(|_| panic!("Expected to find entry {} in queue!", v));
        }
        u
    }

    fn best(&self) -> Node {
        *self
            .queue
            .peek()
            .expect("Expected queue to be non-empty!")
            .0
    }
}

/// Returns a heuristic solution to the DFVS-Problem as a vector of nodes
pub fn greedy_dfvs<S: Selector<G>, T: FromIterator<Node>, G: AdjacencyListIn + GraphEdgeEditing>(
    graph: G,
) -> T {
    let selector = S::new(graph);
    let iter = SelectorIterator::new(selector);
    iter.collect()
}

#[cfg(test)]
pub mod tests {
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::{GraphEdgeEditing, Traversal};
    use crate::heuristics::greedy::{greedy_dfvs, MaxDegreeSelector};

    #[test]
    fn max_degree() {
        let mut org_graph =
            AdjArrayIn::from(&[(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (1, 0), (4, 1)]);
        assert!(!org_graph.is_acyclic());
        let dfvs: Vec<_> = greedy_dfvs::<MaxDegreeSelector<_>, _, _>(org_graph.clone());
        for u in &dfvs {
            org_graph.remove_edges_at_node(*u);
        }
        assert!(org_graph.is_acyclic());
        assert_eq!(dfvs.len(), 1);
        assert_eq!(dfvs[0], 3);
    }
}
