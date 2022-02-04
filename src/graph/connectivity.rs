use super::traversal::*;
use super::*;
use std::cmp::min;

pub trait Connectivity: AdjacencyList + Traversal + Sized {
    /// Returns the strongly connected components of the graph as a Vec<Vec<Node>>
    fn strongly_connected_components(&self) -> Vec<Vec<Node>> {
        let sc = StronglyConnected::new(self);
        sc.find()
    }

    /// Returns the strongly connected components of the graph as a Vec<Vec<Node>>
    /// In contrast to [strongly_connected_components], this methods includes SCCs of size 1
    /// if and only if the node has a self-loop
    fn strongly_connected_components_no_singletons(&self) -> Vec<Vec<Node>> {
        let mut sc = StronglyConnected::new(self);
        sc.set_include_singletons(false);
        sc.find()
    }
}

impl<T: AdjacencyList + Traversal + Sized> Connectivity for T {}

pub struct StronglyConnected<'a, T: AdjacencyList> {
    graph: &'a T,
    idx: Node,
    stack: UniqueNodeStack,
    indices: Vec<Option<Node>>,
    low_links: Vec<Node>,
    components: Vec<Vec<Node>>,
    include_singletons: bool,
}

impl<'a, T: AdjacencyList> StronglyConnected<'a, T> {
    pub fn new(graph: &'a T) -> Self {
        Self {
            graph,
            idx: 0,
            stack: UniqueNodeStack::new(graph.len()),
            indices: vec![None; graph.len()],
            low_links: vec![0; graph.len()],
            components: vec![],
            include_singletons: true,
        }
    }

    pub fn find(mut self) -> Vec<Vec<Node>> {
        for v in self.graph.vertices() {
            if self.indices[v as usize].is_none() {
                self.sc(v);
            }
        }

        debug_assert!(self.stack.is_empty());

        self.components
    }

    pub fn set_include_singletons(&mut self, include: bool) {
        self.include_singletons = include;
    }

    fn sc(&mut self, v: Node) {
        self.indices[v as usize] = Some(self.idx);
        self.low_links[v as usize] = self.idx;
        self.idx += 1;

        let initial_stack_len = self.stack.len();
        self.stack.try_push(v);

        let mut self_loop = false;

        for w in self.graph.out_neighbors(v) {
            self_loop |= w == v;

            if self.indices[w as usize].is_none() {
                self.sc(w);
                self.low_links[v as usize] =
                    min(self.low_links[v as usize], self.low_links[w as usize]);
            } else if self.stack.contains(w) {
                self.low_links[v as usize] = min(
                    self.low_links[v as usize],
                    self.indices[w as usize].unwrap(),
                );
            }
        }

        if self.low_links[v as usize] == self.indices[v as usize].unwrap() {
            if !self.include_singletons && self.stack.peek().unwrap() == v && !self_loop {
                // skip producing component descriptor, since we have a singleton node
                // but we need to undo
                self.stack.pop();
            } else {
                // this component goes into the result, so produce a descriptor and clean-up stack
                // while doing so
                let mut component = Vec::with_capacity(self.stack.len() - initial_stack_len);
                loop {
                    let w = self.stack.pop().unwrap();
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                self.components.push(component);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn sort_sccs(mut sccs: Vec<Vec<Node>>) -> Vec<Vec<Node>> {
        for scc in &mut sccs {
            scc.sort();
        }
        sccs.sort_by(|a, b| a[0].cmp(&b[0]));
        sccs
    }

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
}
