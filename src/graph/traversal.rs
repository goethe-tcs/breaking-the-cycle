use super::*;
use crate::bitset::BitSet;
use std::collections::VecDeque;

pub trait WithGraphRef<G> {
    fn graph(&self) -> &G;
}

pub trait TraversalState {
    fn visited(&self) -> &BitSet;

    fn did_visit_node(&self, u: Node) -> bool {
        self.visited()[u as usize]
    }
}

pub trait NodeSequencer {
    // would prefer this to be private
    fn init(u: Node) -> Self;
    fn push(&mut self, u: Node);
    fn pop(&mut self) -> Option<Node>;
    fn cardinality(&self) -> usize;
}

impl NodeSequencer for VecDeque<Node> {
    fn init(u: Node) -> Self {
        Self::from(vec![u])
    }
    fn push(&mut self, u: Node) {
        self.push_back(u)
    }
    fn pop(&mut self) -> Option<Node> {
        self.pop_front()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

impl NodeSequencer for Vec<Node> {
    fn init(u: Node) -> Self {
        vec![u]
    }
    fn push(&mut self, u: Node) {
        self.push(u)
    }
    fn pop(&mut self) -> Option<Node> {
        self.pop()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

////////////////////////////////////////////////////////////////////////////////////////// BFS & DFS
pub struct TraversalSearch<'a, G: AdjacencyList, S: NodeSequencer> {
    // would prefer this to be private
    graph: &'a G,
    visited: BitSet,
    sequencer: S,
    pre_push: Option<Box<dyn FnMut(Node, Node) + 'a>>,
}

pub type BFS<'a, G> = TraversalSearch<'a, G, VecDeque<Node>>;
pub type DFS<'a, G> = TraversalSearch<'a, G, Vec<Node>>;

impl<'a, G: AdjacencyList, S: NodeSequencer> WithGraphRef<G> for TraversalSearch<'a, G, S> {
    fn graph(&self) -> &G {
        self.graph
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer> TraversalState for TraversalSearch<'a, G, S> {
    fn visited(&self) -> &BitSet {
        &self.visited
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer> Iterator for TraversalSearch<'a, G, S> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        let u = self.sequencer.pop()?;

        for v in self.graph.out_neighbors(u) {
            if !self.visited[v as usize] {
                if let Some(f) = &mut self.pre_push {
                    f(u, v);
                }
                self.sequencer.push(v);
                self.visited.set_bit(v as usize);
            }
        }

        Some(u)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.sequencer.cardinality(),
            Some(self.graph.len() - self.visited.cardinality()),
        )
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer> TraversalSearch<'a, G, S> {
    pub fn new(graph: &'a G, start: Node) -> Self {
        let mut visited = BitSet::new(graph.len());
        visited.set_bit(start as usize);
        Self {
            graph,
            visited,
            sequencer: S::init(start),
            pre_push: None,
        }
    }

    /// Tries to restart the search at an yet unvisited node and returns
    /// true iff successful. Requires that search came to a hold earlier,
    /// i.e. self.next() returned None
    pub fn try_restart_at_unvisited(&mut self) -> bool {
        assert_eq!(self.sequencer.cardinality(), 0);
        match self.visited.get_first_unset() {
            None => false,
            Some(x) => {
                self.visited.set_bit(x);
                self.sequencer.push(x as Node);
                true
            }
        }
    }

    // Registers a boxed dynamic Trait Object implementing Fn(Node, Node)->().
    // It is called every time a new vertex is pushed on the sequencer.
    // The first argument is the vertex popped from the sequencer, and the second the vertex being pushed.
    pub fn register_pre_push(mut self, f: Box<dyn FnMut(Node, Node) + 'a>) -> Self {
        self.pre_push = Some(f);
        self
    }
}

///////////////////////////////////////////////////////////////////////////////////////// TopoSearch
pub struct TopoSearch<'a, G> {
    graph: &'a G,
    in_degs: Vec<Node>,
    stack: Vec<Node>,
}

impl<'a, G: AdjacencyList> WithGraphRef<G> for TopoSearch<'a, G> {
    fn graph(&self) -> &G {
        self.graph
    }
}

impl<'a, G: AdjacencyList> Iterator for TopoSearch<'a, G> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        let u = self.stack.pop()?;

        for v in self.graph.out_neighbors(u) {
            self.in_degs[v as usize] -= 1;
            if self.in_degs[v as usize] == 0 {
                self.stack.push(v);
            }
        }

        Some(u)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), Some(self.graph.len()))
    }
}

impl<'a, G: AdjacencyList> TopoSearch<'a, G> {
    fn new(graph: &'a G) -> Self {
        // add an in_degree getter to each graph?
        let mut in_degs: Vec<Node> = vec![0; graph.len()];
        for u in graph.vertices() {
            for v in graph.out_neighbors(u) {
                // u -> v
                in_degs[v as usize] += 1;
            }
        }

        let stack: Vec<Node> = in_degs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| if *d == 0 { Some(i as Node) } else { None })
            .collect();

        Self {
            graph,
            in_degs,
            stack,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////// Convenience
pub trait RankFromOrder<'a, G: 'a + AdjacencyList>:
    WithGraphRef<G> + Iterator<Item = Node> + Sized
{
    /// Consumes a graph traversal iterator and returns a mapping, where the i-th
    /// item contains the rank (starting from 0) as which it was iterated over.
    /// Returns None iff not all nodes were iterated
    fn ranking(mut self) -> Option<Vec<Node>> {
        let mut ranking = vec![Node::MAX; self.graph().len()];
        let mut rank: Node = 0;

        for u in self.by_ref() {
            assert_eq!(ranking[u as usize], Node::MAX); // assert no item is repeated by iterator
            ranking[u as usize] = rank;
            rank += 1;
        }

        if rank == self.graph().number_of_nodes() {
            Some(ranking)
        } else {
            None
        }
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer> RankFromOrder<'a, G> for TraversalSearch<'a, G, S> {}

impl<'a, G: AdjacencyList> RankFromOrder<'a, G> for TopoSearch<'a, G> {}

/// Offers graph traversal algorithms as methods of the graph representation
pub trait Traversal: AdjacencyList + Sized {
    /// Returns an iterator traversing nodes reachable from `start` in breadth-first-search order
    fn bfs(&self, start: Node) -> BFS<Self> {
        BFS::new(self, start)
    }

    /// Returns an iterator traversing nodes reachable from `start` in depth-first-search order
    fn dfs(&self, start: Node) -> DFS<Self> {
        DFS::new(self, start)
    }

    /// Returns an iterator traversing nodes in acyclic order. The iterator stops prematurely
    /// iff the graph is not acyclic (see `is_acyclic`)
    fn topo_search(&self) -> TopoSearch<Self> {
        TopoSearch::new(self)
    }

    /// Returns true iff the graph is acyclic, i.e. there exists a order f of nodes, such that
    /// for all edge (u, v) we have f(u) < f(v)
    fn is_acyclic(&self) -> bool {
        self.topo_search().count() == self.len()
    }
}

impl<T: AdjacencyList + Sized> Traversal for T {}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_bfs_order() {
        //  / 2 --- \
        // 1         4 - 3
        //  \ 0 - 5 /
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);

        {
            let order: Vec<Node> = graph.bfs(1).collect();
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);
            assert!((order[1] == 0 && order[2] == 2) || (order[2] == 0 && order[1] == 2));
            assert!((order[3] == 4 && order[4] == 5) || (order[4] == 4 && order[3] == 5));
            assert_eq!(order[5], 3);
        }

        {
            let order: Vec<Node> = BFS::new(&graph, 5).collect();
            assert_eq!(order, [5, 4, 3]);
        }
    }

    #[test]
    fn test_dfs_order() {
        //  / 2
        // 1         4 - 3
        //  \ 0 - 5 /
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);

        {
            let order: Vec<Node> = DFS::new(&graph, 1).collect();
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);

            if order[1] == 2 {
                assert_eq!(order[2..6], [0, 5, 4, 3]);
            } else {
                assert_eq!(order[1..6], [0, 5, 4, 3, 2]);
            }
        }

        {
            let order: Vec<Node> = graph.dfs(5).collect();
            assert_eq!(order, [5, 4, 3]);
        }
    }

    #[test]
    fn test_topology_rank() {
        let mut graph = AdjListMatrix::from(&[(2, 0), (1, 0), (0, 3), (0, 4), (0, 5), (3, 6)]);

        {
            let ranks = graph.topo_search().ranking().unwrap();
            assert_eq!(*ranks.iter().min().unwrap(), 0);
            assert_eq!(*ranks.iter().max().unwrap(), graph.number_of_nodes() - 1);
            for (u, v) in graph.edges() {
                assert!(ranks[u as usize] < ranks[v as usize]);
            }
        }

        graph.add_edge(6, 2); // introduce cycle
        {
            let topo = graph.topo_search().ranking();
            assert!(topo.is_none());
        }
    }

    #[test]
    fn test_is_acyclic() {
        let mut graph = AdjListMatrix::from(&[(2, 0), (1, 0), (0, 3), (0, 4), (0, 5), (3, 6)]);
        assert!(graph.is_acyclic());
        graph.add_edge(6, 2); // introduce cycle
        assert!(!graph.is_acyclic());
    }
}
