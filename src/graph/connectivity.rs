use crate::bitset::BitSet;
use std::cmp::min;
use super::*;
use super::traversal::*;

pub trait Connectivity : AdjacencyList + Traversal + Sized {
    /// Returns the strongly connected components of the graph as a Vec<Vec<Node>>
    fn strongly_connected_components(&self) -> Vec<Vec<Node>> {
        let sc = StronglyConnected::new(self);
        sc.find()
    }

    /// Returns the connected components of the graph as BitSets
    fn connected_components(&self) -> Vec<BitSet> {
        if self.len() == 0 {
            // questionable corner case: what is a CC on an empty graph?
            return Vec::new();
        }

        let mut components: Vec<BitSet> = vec![];
        let mut traversal = self.dfs_undirected(0);

        loop {
            let mut component = BitSet::new(self.len());
            for x in traversal.by_ref() {
                component.set_bit(x as usize);
            }
            components.push(component);

            if !traversal.try_restart_at_unvisited() {
                break;
            }
        }

        components
    }
}

impl<T : AdjacencyList + Traversal + Sized> Connectivity for T {}

pub struct StronglyConnected<'a, T : AdjacencyList> {
    graph: &'a T,
    idx: Node,
    stack: Vec<Node>,
    indices: Vec<Option<Node>>,
    low_links: Vec<Node>,
    on_stack: BitSet,
    components: Vec<Vec<Node>>,
}

impl<'a, T : AdjacencyList> StronglyConnected<'a, T> {
    pub fn new(graph: &'a T) -> Self {
        Self {
            graph,
            idx: 0,
            stack: vec![],
            indices: vec![None; graph.len()],
            low_links: vec![0; graph.len()],
            on_stack: BitSet::new(graph.len()),
            components: vec![],
        }
    }

    pub fn find(mut self) -> Vec<Vec<Node>> {
        for v in self.graph.vertices() {
            if self.indices[v as usize].is_none() {
                self.sc(v);
            }
        }
        self.components
    }

    fn sc(&mut self, v: Node) {
        self.indices[v as usize] = Some(self.idx);
        self.low_links[v as usize] = self.idx;
        self.idx += 1;
        self.stack.push(v);
        self.on_stack.set_bit(v as usize);

        for w in self.graph.out_neighbors(v) {
            if self.indices[w as usize].is_none() {
                self.sc(w);
                self.low_links[v as usize] =
                    min(self.low_links[v as usize], self.low_links[w as usize]);
            } else if self.on_stack[w as usize] {
                self.low_links[v as usize] = min(
                    self.low_links[v as usize],
                    self.indices[w as usize].unwrap(),
                );
            }
        }

        if self.low_links[v as usize] == self.indices[v as usize].unwrap() {
            // found SC
            let mut component = Vec::with_capacity(self.graph.len());
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack.unset_bit(w as usize);
                component.push(w);
                if w == v {
                    break;
                }
            }
            self.components.push(component);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    pub fn cc() {
        let mut graph = AdjListMatrix::new(9);
        graph.add_edges(&[
            (0, 3), (6, 3), (6, 7),
            (4, 1), (1, 8),
        ]);
        let mut ccs = graph.connected_components();

        assert_eq!(ccs.len(), 4);

        assert!(!ccs[0].is_empty());
        assert!(!ccs[1].is_empty());
        assert!(!ccs[2].is_empty());
        assert!(!ccs[3].is_empty());

        ccs.sort_by(|a,b| {a.get_first_set().unwrap().cmp(&b.get_first_set().unwrap())});

        assert_eq!(ccs[0].to_vec(), [0,3,6,7]);
        assert_eq!(ccs[1].to_vec(), [1,4,8]);
        assert_eq!(ccs[2].to_vec(), [2]);
        assert_eq!(ccs[3].to_vec(), [5]);
    }

    #[test]
    pub fn cc_empty() {
        let graph = AdjListMatrix::new(0);
        let ccs = graph.connected_components();

        assert_eq!(ccs.len(), 0);
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

        let mut sccs = graph.strongly_connected_components();
        assert_eq!(sccs.len(), 3);
        assert!(!sccs[0].is_empty());
        assert!(!sccs[1].is_empty());
        assert!(!sccs[2].is_empty());

        for scc in &mut sccs {
            scc.sort();
        }
        sccs.sort_by(|a, b| a[0].cmp(&b[0]));
        assert_eq!(sccs[0], [0, 1, 4]);
        assert_eq!(sccs[1], [2, 3, 7]);
        assert_eq!(sccs[2], [5, 6]);
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
