use super::*;
use itertools::Itertools;
use std::iter::FusedIterator;

pub trait Connectivity: AdjacencyList + Traversal + Sized {
    /// Returns the strongly connected components of the graph as a Vec<Vec<Node>>
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::*;
    /// // {0,1} and {4,5} are scc pairs, 2 is a loop, 3 is a singleton
    /// let graph = AdjListMatrix::from(&[(0, 1), (1, 0), (2, 2), (4, 5), (5, 4),]);
    /// let sccs = graph.strongly_connected_components();
    /// let sccs = dfvs::graph::connectivity::sort_sccs(sccs); // return order is non-deterministic, so we sort here
    /// assert_eq!(sccs[0], [0, 1]);
    /// assert_eq!(sccs[1], [2]);
    /// assert_eq!(sccs[2], [3]);
    /// assert_eq!(sccs[3], [4, 5]);
    /// ```
    fn strongly_connected_components(&self) -> Vec<Vec<Node>> {
        let sc = StronglyConnected::new(self);
        sc.collect_vec()
    }

    /// Returns the strongly connected components of the graph as a Vec<Vec<Node>>
    /// In contrast to [strongly_connected_components], this methods includes SCCs of size 1
    /// if and only if the node has a self-loop
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::*;
    /// // {0,1} and {4,5} are scc pairs, 2 is a loop, 3 is a singleton
    /// let graph = AdjListMatrix::from(&[(0, 1), (1, 0), (2, 2), (4, 5), (5, 4),]);
    /// let sccs = graph.strongly_connected_components_no_singletons();
    /// let sccs = dfvs::graph::connectivity::sort_sccs(sccs); // return order is non-deterministic, so we sort here
    /// assert_eq!(sccs[0], [0, 1]);
    /// assert_eq!(sccs[1], [2]);
    /// // in contrast to [strongly_connected_components] node 3 is not returned!
    /// assert_eq!(sccs[2], [4, 5]);
    /// ```
    fn strongly_connected_components_no_singletons(&self) -> Vec<Vec<Node>> {
        let mut sc = StronglyConnected::new(self);
        sc.set_include_singletons(false);
        sc.collect_vec()
    }
}

impl<T: AdjacencyList + Sized> Connectivity for T {}

/// Implementation of Tarjan's Algorithm for Strongly Connected Components.
/// It is designed as an iterator that emits the nodes of one strongly connected component at a
/// time. Observe that the order of nodes within a component is non-deterministic; the order of the
/// components themselves are in the reverse topological order of the SCCs (i.e. if each SCC
/// were contracted into a single node).
pub struct StronglyConnected<'a, T: AdjacencyList> {
    graph: &'a T,
    idx: Node,

    states: Vec<NodeState>,
    potentially_unvisited: usize,

    include_singletons: bool,

    path_stack: Vec<Node>,

    call_stack: Vec<StackFrame<'a, T>>,
}

impl<'a, T: AdjacencyList> StronglyConnected<'a, T> {
    /// Construct the iterator for some graph
    pub fn new(graph: &'a T) -> Self {
        Self {
            graph,
            idx: 0,
            states: vec![Default::default(); graph.len()],
            potentially_unvisited: 0,

            include_singletons: true,

            path_stack: Vec::with_capacity(32),
            call_stack: Vec::with_capacity(32),
        }
    }

    /// Each node that is not part of a circle is returned as its own SCC.
    /// By setting `include = false`, those nodes are not returned (which can lead to a significant
    /// performance boost)
    pub fn set_include_singletons(&mut self, include: bool) {
        self.include_singletons = include;
    }

    /// Just like in a classic DFS where we want to compute a spanning-forest, we will need to
    /// to visit each node at least once. We start we node 0, and cover all nodes reachable from
    /// there in `search`. Then, we search for an untouched node here, and start over.
    fn next_unvisited_node(&mut self) -> Option<Node> {
        while self.potentially_unvisited < self.graph.len() {
            if !self.states[self.potentially_unvisited].visited {
                let v = self.potentially_unvisited as Node;
                self.push_node(v, None);
                return Some(v);
            }

            self.potentially_unvisited += 1;
        }
        None
    }

    /// Put a pristine stack frame on the call stack. Roughly speaking, this is the first step
    /// to a recursive call of search.
    fn push_node(&mut self, node: Node, parent: Option<Node>) {
        self.call_stack.push(StackFrame {
            node,
            parent: parent.unwrap_or(node),
            initial_stack_len: 0,
            first_call: true,
            has_loop: false,
            neighbors: self.graph.out_neighbors(node),
        });
    }

    fn search(&mut self) -> Option<Vec<Node>> {
        /*
        Tarjan's algorithm is typically described in a recursive fashion similarly to DFS
        with some extra steps. This design has two issues:
         1.) We cannot easily build an iterator from it
         2.) For large graphs we get stack overflows.

        To overcome these issues, we use the explicit call stack `self.call_stack` that simulates
        recursive calls. On first visit of a node v it is assigned a "DFS rank"ish index and
        additionally the same low_link value. This value stores the smallest known index of any node known to be
        reachable from v. We then process all of its neighbors (which may trigger recursive calls).
        Eventually, all nodes in an SCC will have the same low_link and the unique node with this
        index becomes the arbitrary representative of this SCC (known as root).

        The key design is that the whole computation is wrapped in a `while` loop and all state
        (including iterators) is stored in `self.call_stack`. So we continue execution directly with
        another iteration. Alternative, we can pause processing, return an value and resume by
        reentering the function.
        */

        'recurse: while let Some(frame) = self.call_stack.last_mut() {
            let v = frame.node;

            if frame.first_call {
                frame.first_call = false;
                frame.initial_stack_len = self.path_stack.len() as Node;

                self.states[v as usize].visit(self.idx);
                self.idx += 1;

                self.path_stack.push(v);
            }

            for w in frame.neighbors.by_ref() {
                let w_state = self.states[w as usize];
                frame.has_loop |= w == v;

                if !w_state.visited {
                    self.push_node(w, Some(v));
                    continue 'recurse;
                } else if w_state.on_stack {
                    self.states[frame.node as usize].try_lower_link(w_state.index);
                }
            }

            let frame = self.call_stack.pop().unwrap();
            let state = self.states[v as usize];

            self.states[frame.parent as usize].try_lower_link(state.low_links);

            if state.is_root() {
                if !self.include_singletons
                    && *self.path_stack.last().unwrap() == v
                    && !frame.has_loop
                {
                    // skip producing component descriptor, since we have a singleton node
                    // but we need to undo
                    self.states[v as usize].on_stack = false;
                    self.path_stack.pop();
                } else {
                    // this component goes into the result, so produce a descriptor and clean-up stack
                    // while doing so
                    let component = self.path_stack
                        [frame.initial_stack_len as usize..self.path_stack.len()]
                        .iter()
                        .copied()
                        .collect_vec();

                    self.path_stack.truncate(frame.initial_stack_len as usize);

                    for &w in &component {
                        self.states[w as usize].on_stack = false;
                    }

                    debug_assert_eq!(*component.first().unwrap(), v);

                    return Some(component);
                }
            }
        }

        None
    }
}

impl<'a, T: AdjacencyList> Iterator for StronglyConnected<'a, T> {
    type Item = Vec<Node>;

    /// Returns either a vector of node ids that form an SCC.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.search() {
                Some(x) => {
                    return Some(x);
                }
                None => {
                    self.next_unvisited_node()?;
                }
            }
        }
    }
}

impl<'a, T: AdjacencyList> FusedIterator for StronglyConnected<'a, T> {}

#[derive(Debug, Clone)]
struct StackFrame<'a, T: AdjacencyList + 'a> {
    node: Node,
    parent: Node,
    initial_stack_len: Node,
    first_call: bool,
    has_loop: bool,
    neighbors: T::Iter<'a>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NodeState {
    visited: bool,
    on_stack: bool,
    index: Node,
    low_links: Node,
}

impl NodeState {
    fn visit(&mut self, u: Node) {
        debug_assert!(!self.visited);
        self.index = u;
        self.low_links = u;
        self.visited = true;
        self.on_stack = true;
    }

    fn try_lower_link(&mut self, l: Node) {
        self.low_links = self.low_links.min(l);
    }

    fn is_root(&self) -> bool {
        self.index == self.low_links
    }
}

/// Sorts the nodes in each SCC increasingly and then the SCCs themselves lexicographically.
pub fn sort_sccs(mut sccs: Vec<Vec<Node>>) -> Vec<Vec<Node>> {
    sccs.iter_mut().for_each(|scc| scc.sort_unstable());
    sccs.sort_by(|a, b| a[0].cmp(&b[0]));
    sccs
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;
    use crate::random_models::gnp::generate_gnp;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    pub fn scc() {
        let graph = AdjListMatrix::from(&[
            (0, 1),
            (1, 2),
            (1, 4),
            (1, 5),
            (2, 6),
            (2, 3),
            (3, 2),
            (3, 7),
            (4, 0),
            (4, 5),
            (5, 6),
            (6, 5),
            (7, 3),
            (7, 6),
        ]);

        let sccs = graph.strongly_connected_components();
        assert_eq!(sccs.len(), 3);
        assert!(!sccs[0].is_empty());
        assert!(!sccs[1].is_empty());
        assert!(!sccs[2].is_empty());

        let sccs = sort_sccs(sccs);
        assert_eq!(sccs[0], [0, 1, 4]);
        assert_eq!(sccs[1], [2, 3, 7]);
        assert_eq!(sccs[2], [5, 6]);
    }

    #[test]
    pub fn scc_singletons() {
        // {0,1} and {4,5} are scc pairs, 2 is a loop, 3 is a singleton
        let graph = AdjListMatrix::from(&[
            (0, 1),
            (1, 0),
            (2, 2),
            // 3 is missing
            (4, 5),
            (5, 4),
        ]);

        {
            let sccs = graph.strongly_connected_components();
            assert_eq!(sccs.len(), 4);

            let sccs = sort_sccs(sccs);
            assert_eq!(sccs[0], [0, 1]);
            assert_eq!(sccs[1], [2]);
            assert_eq!(sccs[2], [3]); // 3 is included
            assert_eq!(sccs[3], [4, 5]);
        }

        {
            let sccs = graph.strongly_connected_components_no_singletons();
            assert_eq!(sccs.len(), 3);
            let sccs = sort_sccs(sccs);

            assert_eq!(sccs[0], [0, 1]);
            assert_eq!(sccs[1], [2]);
            assert_eq!(sccs[2], [4, 5]);
        }
    }

    #[test]
    pub fn scc_tree() {
        let graph = AdjListMatrix::from(&[(0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (3, 6)]);

        let mut sccs = graph.strongly_connected_components();
        // in a directed tree each vertex is a strongly connected component
        assert_eq!(sccs.len(), 7);

        sccs.sort_by(|a, b| a[0].cmp(&b[0]));
        for (i, scc) in sccs.iter().enumerate() {
            assert_eq!(i as Node, scc[0]);
        }
    }

    #[test]
    fn scc_gnp() {
        let mut gen = Pcg64::seed_from_u64(1234);

        for i in 0..10 {
            let n = 10000;
            let graph: AdjListMatrix = generate_gnp(&mut gen, n, 0.5 / (n as f64) * (i as f64));
            assert_eq!(
                StronglyConnected::new(&graph)
                    .map(|x| x.len())
                    .sum::<usize>(),
                n as usize
            );
        }
    }

    #[test]
    fn scc_long_cycle() {
        // assert that we can deal with very deep stacks
        let n: Node = 10_000;
        let mut graph = AdjListMatrix::new(n as usize);
        graph.connect_cycle((0..n).into_iter());
        let sccs = graph.strongly_connected_components();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs.first().unwrap().len(), n as usize);
    }

    #[bench]
    fn bench_tarjan_sparse(b: &mut test::Bencher) {
        let mut gen = Pcg64::seed_from_u64(1234);
        let graph: AdjListMatrix = generate_gnp(&mut gen, 10000, 0.0001);

        b.iter(|| {
            let sc = StronglyConnected::new(&graph);
            sc.map(|x| x.len()).sum::<usize>()
        });
    }

    #[bench]
    fn bench_tarjan_dense(b: &mut test::Bencher) {
        let mut gen = Pcg64::seed_from_u64(1234);
        let graph: AdjListMatrix = generate_gnp(&mut gen, 10000, 0.001);

        b.iter(|| {
            let sc = StronglyConnected::new(&graph);
            sc.map(|x| x.len()).sum::<usize>()
        });
    }
}
