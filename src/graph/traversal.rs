use super::*;
use crate::bitset::BitSet;
use std::collections::VecDeque;
use std::marker::PhantomData;

pub trait WithGraphRef<G> {
    fn graph(&self) -> &G;
}

pub trait TraversalState {
    fn visited(&self) -> &BitSet;

    fn did_visit_node(&self, u: Node) -> bool {
        self.visited()[u as usize]
    }
}

pub trait SequencedItem {
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self;
    fn new_without_predecessor(item: Node) -> Self;
    fn item(&self) -> Node;
    fn predecessor(&self) -> Option<Node>;
    fn predecessor_with_item(&self) -> (Option<Node>, Node) {
        (self.predecessor(), self.item())
    }
}

impl SequencedItem for Node {
    fn new_with_predecessor(_: Node, item: Node) -> Self {
        item
    }
    fn new_without_predecessor(item: Node) -> Self {
        item
    }
    fn item(&self) -> Node {
        *self
    }
    fn predecessor(&self) -> Option<Node> {
        None
    }
}

// We use an ordinary Edge to encode the item and the optional predecessor to safe some
// memory. We can easily accomplish this by exploiting that the traversal algorithms do
// not take self-loops. So "None" is encoded by setting the predecessor as the node itself.
type PredecessorOfNode = Edge;
impl SequencedItem for PredecessorOfNode {
    fn new_with_predecessor(predecessor: Node, item: Node) -> Self {
        (predecessor, item)
    }
    fn new_without_predecessor(item: Node) -> Self {
        (item, item)
    }
    fn item(&self) -> Node {
        self.1
    }
    fn predecessor(&self) -> Option<Node> {
        if self.0 == self.1 {
            None
        } else {
            Some(self.0)
        }
    }
}

pub trait NodeSequencer<T> {
    // would prefer this to be private
    fn init(u: T) -> Self;
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
    fn cardinality(&self) -> usize;
}

impl<T> NodeSequencer<T> for VecDeque<T> {
    fn init(u: T) -> Self {
        Self::from(vec![u])
    }
    fn push(&mut self, u: T) {
        self.push_back(u)
    }
    fn pop(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

impl<T> NodeSequencer<T> for Vec<T> {
    fn init(u: T) -> Self {
        vec![u]
    }
    fn push(&mut self, u: T) {
        self.push(u)
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn cardinality(&self) -> usize {
        self.len()
    }
}

////////////////////////////////////////////////////////////////////////////////////////// BFS & DFS
pub struct TraversalSearch<'a, G: AdjacencyList, S: NodeSequencer<I>, I: SequencedItem> {
    // would prefer this to be private
    graph: &'a G,
    visited: BitSet,
    sequencer: S,
    pre_push: Option<Box<dyn FnMut(Node, Node) + 'a>>,
    _item: PhantomData<I>,
}

pub type BFS<'a, G> = TraversalSearch<'a, G, VecDeque<Node>, Node>;
pub type DFS<'a, G> = TraversalSearch<'a, G, Vec<Node>, Node>;
pub type BFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, VecDeque<PredecessorOfNode>, PredecessorOfNode>;
pub type DFSWithPredecessor<'a, G> =
    TraversalSearch<'a, G, Vec<PredecessorOfNode>, PredecessorOfNode>;

impl<'a, G: AdjacencyList, S: NodeSequencer<I>, I: SequencedItem> WithGraphRef<G>
    for TraversalSearch<'a, G, S, I>
{
    fn graph(&self) -> &G {
        self.graph
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer<I>, I: SequencedItem> TraversalState
    for TraversalSearch<'a, G, S, I>
{
    fn visited(&self) -> &BitSet {
        &self.visited
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer<I>, I: SequencedItem> Iterator
    for TraversalSearch<'a, G, S, I>
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        let popped = self.sequencer.pop()?;
        let u = popped.item();

        for v in self.graph.out_neighbors(u) {
            if !self.visited[v as usize] {
                if let Some(f) = &mut self.pre_push {
                    f(u, v);
                }
                self.sequencer.push(I::new_with_predecessor(u, v));
                self.visited.set_bit(v as usize);
            }
        }

        Some(popped)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.sequencer.cardinality(),
            Some(self.graph.len() - self.visited.cardinality()),
        )
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer<I>, I: SequencedItem> TraversalSearch<'a, G, S, I> {
    pub fn new(graph: &'a G, start: Node) -> Self {
        let mut visited = BitSet::new(graph.len());
        visited.set_bit(start as usize);
        Self {
            graph,
            visited,
            sequencer: S::init(I::new_without_predecessor(start)),
            pre_push: None,
            _item: PhantomData,
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
                self.sequencer.push(I::new_without_predecessor(x as Node));
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

impl<'a, G: AdjacencyList, S: NodeSequencer<Node>> RankFromOrder<'a, G>
    for TraversalSearch<'a, G, S, Node>
{
}

impl<'a, G: AdjacencyList> RankFromOrder<'a, G> for TopoSearch<'a, G> {}

pub trait TraversalTree<'a, G: 'a + AdjacencyList>:
    WithGraphRef<G> + Iterator<Item = PredecessorOfNode> + Sized
{
    /// Consumes the underlying graph traversal iterator and records the implied tree structure
    /// into an parent-array, i.e. [result\[i\]] stores the predecessor of node [i]. It is the
    /// calling code's responsibility to ensure that the slice [tree] is sufficiently large to
    /// store all reachable nodes (i.e. in general of size at least [graph.len()]).
    fn parent_array_into(&mut self, tree: &mut [Node]) {
        for pred_with_item in self.by_ref() {
            if let Some(p) = pred_with_item.predecessor() {
                tree[pred_with_item.item() as usize] = p;
            }
        }
    }

    /// Calls allocates a vector of size [graph.len()] and calls [self.parent_array_into] on it.
    /// Unvisited nodes have themselves as parents.
    fn parent_array(&mut self) -> Vec<Node> {
        let mut tree: Vec<_> = self.graph().vertices_range().collect();
        self.parent_array_into(&mut tree);
        tree
    }
}

impl<'a, G: AdjacencyList, S: NodeSequencer<PredecessorOfNode>> TraversalTree<'a, G>
    for TraversalSearch<'a, G, S, PredecessorOfNode>
{
}

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

    /// Returns an iterator traversing nodes reachable from `start` in breadth-first-search order
    /// The items returned are the edges taken
    fn bfs_with_predecessor(&self, start: Node) -> BFSWithPredecessor<Self> {
        BFSWithPredecessor::new(self, start)
    }

    /// Returns an iterator traversing nodes reachable from `start` in depth-first-search order
    /// The items returned are the edges taken
    fn dfs_with_predecessor(&self, start: Node) -> DFSWithPredecessor<Self> {
        DFSWithPredecessor::new(self, start)
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
    fn test_bfs_with_predecessor() {
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);

        let mut edges: Vec<_> = graph
            .bfs_with_predecessor(1)
            .map(|x| x.predecessor_with_item())
            .collect();
        edges.sort();
        assert_eq!(
            edges,
            vec![
                (None, 1),
                (Some(0), 5),
                (Some(1), 0),
                (Some(1), 2),
                (Some(2), 4),
                (Some(4), 3)
            ]
        );
    }

    #[test]
    fn test_bfs_tree() {
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (2, 4), (5, 4)]);
        let tree = graph.bfs_with_predecessor(1).parent_array();
        assert_eq!(tree, vec![1, 1, 1, 4, 2, 0]);
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
    fn test_dfs_tree() {
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);
        let tree = graph.dfs_with_predecessor(1).parent_array();
        assert_eq!(tree, vec![1, 1, 1, 4, 5, 0]);
    }

    #[test]
    fn test_dfs_with_predecessor() {
        let graph = AdjListMatrix::from(&[(1, 2), (1, 0), (4, 3), (0, 5), (5, 4)]);

        let mut edges: Vec<_> = graph
            .dfs_with_predecessor(1)
            .map(|x| x.predecessor_with_item())
            .collect();
        edges.sort();
        assert_eq!(
            edges,
            vec![
                (None, 1),
                (Some(0), 5),
                (Some(1), 0),
                (Some(1), 2),
                (Some(4), 3),
                (Some(5), 4)
            ]
        );
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
