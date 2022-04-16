use super::io::DotWrite;
use super::*;
use itertools::Itertools;
use std::fmt::Debug;
use std::{fmt, str};

#[derive(Clone, Default)]
pub struct AdjArrayUndir {
    neighbors: Vec<Neighborhood>,
    m: usize,
}

impl GraphNew for AdjArrayUndir {
    fn new(n: usize) -> Self {
        Self {
            neighbors: vec![Neighborhood::new(); n],
            m: 0,
        }
    }
}
graph_macros::impl_helper_graph_from_edges!(AdjArrayUndir);
graph_macros::impl_helper_graph_debug!(AdjArrayUndir);

impl GraphOrder for AdjArrayUndir {
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> Node {
        self.neighbors.len() as Node
    }

    fn number_of_edges(&self) -> usize {
        self.m
    }

    fn len(&self) -> usize {
        self.neighbors.len()
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
    }
}

impl AdjacencyList for AdjArrayUndir {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.neighbors[u as usize]
            .out_and_undir_neighbors()
            .iter()
            .copied()
    }

    fn out_degree(&self, u: Node) -> Node {
        self.neighbors[u as usize].out_and_undir_neighbors().len() as Node
    }
}

impl AdjacencyListIn for AdjArrayUndir {
    type IterIn<'a> = impl Iterator<Item = Node> + 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_> {
        self.neighbors[u as usize]
            .in_and_undir_neighbors()
            .iter()
            .copied()
    }

    fn in_degree(&self, u: Node) -> Node {
        self.neighbors[u as usize].in_and_undir_neighbors().len() as Node
    }
}

impl AdjacencyListUndir for AdjArrayUndir {
    type IterUndir<'a> = impl Iterator<Item = Node> + 'a;

    fn undir_neighbors(&self, u: Node) -> Self::IterUndir<'_> {
        self.neighbors[u as usize].undir_neighbors().iter().copied()
    }

    fn undir_degree(&self, u: Node) -> Node {
        self.neighbors[u as usize].undir_neighbors().len() as Node
    }
}

impl AdjacencyTest for AdjArrayUndir {
    fn has_edge(&self, u: Node, v: Node) -> bool {
        if u == v {
            self.undir_neighbors(u).contains(&u)
        } else {
            self.out_neighbors(u).contains(&v)
        }
    }
}

impl GraphEdgeEditing for AdjArrayUndir {
    fn add_edge(&mut self, u: Node, v: Node) {
        if u == v {
            self.neighbors[u as usize].undir_add(u);
        } else {
            self.neighbors[v as usize].in_add(u);
            self.neighbors[u as usize].out_add(v);
        }
        self.m += 1;
    }

    fn try_add_edge(&mut self, u: Node, v: Node) -> bool {
        if self.has_edge(u, v) {
            false
        } else {
            self.add_edge(u, v);
            true
        }
    }

    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool {
        let success = if u == v {
            self.neighbors[u as usize].try_undir_remove(u)
        } else if !self.neighbors[u as usize].try_out_remove(v) {
            debug_assert!(!self.neighbors[v as usize].try_in_remove(u));
            false
        } else {
            let res = self.neighbors[v as usize].try_in_remove(u);
            debug_assert!(res);
            true
        };

        self.m -= success as usize;
        success
    }

    fn remove_edges_into_node(&mut self, u: Node) {
        // need to temporarily take out the neighborhood of u to work around the borrow checker
        let mut neighs = std::mem::take(&mut self.neighbors[u as usize]);

        self.m -= neighs.try_undir_remove(u) as usize;
        self.m -= neighs.in_and_undir_neighbors().len();

        for &node in neighs.in_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_out_remove(u);
            debug_assert!(res);
        }

        neighs.in_clear();
        self.neighbors[u as usize] = neighs;
    }

    fn remove_edges_out_of_node(&mut self, u: Node) {
        // need to temporarily take out the neighborhood of u to work around the borrow checker
        let mut neighs = std::mem::take(&mut self.neighbors[u as usize]);

        self.m -= neighs.try_undir_remove(u) as usize;
        self.m -= neighs.out_and_undir_neighbors().len();

        for &node in neighs.out_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_in_remove(u);
            debug_assert!(res);
        }

        neighs.out_clear();
        self.neighbors[u as usize] = neighs;
    }

    fn remove_edges_at_node(&mut self, u: Node) {
        let mut neighs: Neighborhood = std::mem::take(&mut self.neighbors[u as usize]);

        let has_loop = neighs.try_undir_remove(u);
        self.m -= neighs.out_and_undir_neighbors().len();
        self.m -= neighs.in_and_undir_neighbors().len();
        self.m -= has_loop as usize;

        for &node in neighs.in_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_out_remove(u);
            debug_assert!(res);
        }

        for &node in neighs.out_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_in_remove(u);
            debug_assert!(res);
        }
    }

    fn contract_node(&mut self, u: Node) -> Vec<Node> {
        let mut neighs: Neighborhood = std::mem::take(&mut self.neighbors[u as usize]);

        let has_loop = neighs.try_undir_remove(u);
        self.m -= neighs.out_and_undir_neighbors().len()
            + neighs.in_and_undir_neighbors().len()
            + has_loop as usize;

        for &node in neighs.in_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_out_remove(u);
            debug_assert!(res);
            for &out in neighs.out_and_undir_neighbors() {
                self.try_add_edge(node, out);
            }
        }

        for &node in neighs.out_and_undir_neighbors() {
            let res = self.neighbors[node as usize].try_in_remove(u);
            debug_assert!(res);
        }

        neighs.undir_neighbors().iter().copied().collect_vec()
    }
}

/// The `Neighborhood` container stores the in/out/undirected neighbors of a node. An undirected
/// neighbor is a neighbor which is at the same time in- and out-neighbor. The items are kept in
/// a single vector which consists of three contiguous groups:
///  Elements 0..num_out keeps the exactly out-neighbors (those which are not in-neighbors)
///  Elements num_out..(len-num_in) are undirected neighbors
///  Elements (len-num_in)..len are exactly in-neighbors (those which are not out-neighbors)
///
/// This design allows us to return out- and in-neighbor slices which both include undirected
/// neighbors. Since we do not guarantee any item order, adding/removing any type of neighbor
/// can be implement in at most 3 shift/swap operations
#[derive(Clone, Default, Debug)]
struct Neighborhood {
    neighbors: Vec<Node>,
    num_out: Node,
    num_in: Node,
}

impl Neighborhood {
    fn new() -> Self {
        Self {
            neighbors: vec![],
            num_in: 0,
            num_out: 0,
        }
    }

    #[allow(unused)]
    fn with_capacity(n: Node) -> Self {
        Self {
            neighbors: Vec::with_capacity(n as usize),
            num_in: 0,
            num_out: 0,
        }
    }

    fn out_only_neighbors(&self) -> &[Node] {
        self.neighbors.as_slice()[0..self.num_out as usize].as_ref()
    }

    fn out_and_undir_neighbors(&self) -> &[Node] {
        self.neighbors.as_slice()[0..self.first_in_index()].as_ref()
    }

    fn in_only_neighbors(&self) -> &[Node] {
        self.neighbors.as_slice()[self.first_in_index() as usize..self.neighbors.len()].as_ref()
    }

    fn in_and_undir_neighbors(&self) -> &[Node] {
        self.neighbors.as_slice()[self.num_out as usize..self.neighbors.len()].as_ref()
    }

    fn undir_neighbors(&self) -> &[Node] {
        self.neighbors.as_slice()[self.num_out as usize..self.first_in_index()].as_ref()
    }

    fn out_add(&mut self, v: Node) {
        let pos = self.in_only_neighbors().iter().position(|&x| x == v);
        let begin_in = self.first_in_index();
        if let Some(pos) = pos {
            self.neighbors.swap(pos + begin_in, begin_in);
            self.num_in -= 1;
        } else if self.neighbors.len() as Node == self.num_out {
            self.neighbors.push(v);
            self.num_out += 1;
        } else {
            if self.num_in != 0 {
                self.neighbors.push(self.neighbors[begin_in]); // copy first item of in
                self.neighbors[begin_in] = self.neighbors[self.num_out as usize];
            // move first undir item to end
            } else {
                self.neighbors.push(self.neighbors[self.num_out as usize]);
            }

            self.neighbors[self.num_out as usize] = v;

            self.num_out += 1;
        }
    }

    fn in_add(&mut self, v: Node) {
        let pos = self.out_only_neighbors().iter().position(|&x| x == v);
        if let Some(pos) = pos {
            self.neighbors.swap(pos, self.num_out as usize - 1);
            self.num_out -= 1;
        } else {
            self.neighbors.push(v);
            self.num_in += 1;
        }
    }

    fn undir_add(&mut self, v: Node) {
        if self.num_in > 0 {
            let begin_in = self.first_in_index();
            self.neighbors.push(self.neighbors[begin_in]);
            self.neighbors[begin_in] = v;
        } else {
            self.neighbors.push(v);
        }
    }

    fn try_out_remove(&mut self, v: Node) -> bool {
        let pos = self.out_only_neighbors().iter().position(|&x| x == v);
        let begin_in = self.first_in_index();

        if let Some(pos) = pos {
            if self.num_out == self.neighbors.len() as Node {
                self.neighbors.swap_remove(pos);
            } else {
                self.neighbors[pos] = self.neighbors[self.num_out as usize - 1];

                if self.num_in == 0 {
                    self.neighbors.swap_remove(self.num_out as usize - 1);
                } else {
                    self.neighbors[self.num_out as usize - 1] = self.neighbors[begin_in - 1];
                    self.neighbors.swap_remove(begin_in - 1);
                }
            }

            self.num_out -= 1;
            return true;
        }

        let pos = self.undir_neighbors().iter().position(|&x| x == v);
        if let Some(pos) = pos {
            // v was an undir neighbor, so we swap it with the last undir element and grow
            // the in- area to the front
            let pos = pos + self.num_out as usize;
            self.neighbors.swap(pos, begin_in - 1);
            self.num_in += 1;
            return true;
        }

        false
    }

    fn try_in_remove(&mut self, v: Node) -> bool {
        let pos = self.in_only_neighbors().iter().position(|&x| x == v);
        if let Some(pos) = pos {
            // v was an in neighbor, so we swap remove it
            let pos = pos + self.first_in_index();
            self.neighbors.swap_remove(pos);
            self.num_in -= 1;
            return true;
        }

        let pos = self.undir_neighbors().iter().position(|&x| x == v);
        if let Some(pos) = pos {
            // v was an undir neighbor, so we swap it to the front of the undir area and
            // grow the out-neighbors by one to engulf it
            let pos = pos + self.num_out as usize;
            self.neighbors.swap(pos, self.num_out as usize);
            self.num_out += 1;
            return true;
        }

        false
    }

    fn try_undir_remove(&mut self, v: Node) -> bool {
        let pos = self.undir_neighbors().iter().position(|&x| x == v);
        let begin_in = self.first_in_index();

        if let Some(pos) = pos {
            let pos = pos + self.num_out as usize;
            self.neighbors[pos] = self.neighbors[begin_in - 1];
            if begin_in < self.neighbors.len() {
                self.neighbors[begin_in - 1] = self.neighbors.pop().unwrap();
            } else {
                self.neighbors.pop();
            }
            true
        } else {
            false
        }
    }

    fn in_clear(&mut self) {
        self.neighbors.truncate(self.first_in_index());
        self.num_in = 0;
        self.num_out = self.neighbors.len() as Node;
    }

    fn out_clear(&mut self) {
        self.neighbors.drain(0..self.num_out as usize);
        self.num_out = 0;
        self.num_in = self.neighbors.len() as Node;
    }

    // internal
    fn first_in_index(&self) -> usize {
        self.neighbors.len() - (self.num_in as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph_macros::*;
    use super::*;

    base_tests_in!(AdjArrayUndir);

    fn sorted_from_iter<I: Iterator<Item = T>, T: Ord>(it: I) -> Vec<T> {
        let mut vec: Vec<_> = it.collect();
        vec.sort_unstable();
        vec
    }

    fn sorted_from_slice<T: Copy + Ord>(sl: &[T]) -> Vec<T> {
        sorted_from_iter(sl.iter().copied())
    }

    fn tripple(ns: &Neighborhood) -> (Vec<Node>, Vec<Node>, Vec<Node>) {
        (
            sorted_from_slice(ns.out_only_neighbors()),
            sorted_from_slice(ns.undir_neighbors()),
            sorted_from_slice(ns.in_only_neighbors()),
        )
    }

    #[test]
    fn out_only() {
        let mut ns = Neighborhood::new();
        ns.out_add(1);
        ns.out_add(2);

        assert_eq!(tripple(&ns), (vec![1, 2], vec![], vec![]));

        assert!(!ns.try_out_remove(3));
        assert!(ns.try_out_remove(1));

        assert_eq!(tripple(&ns), (vec![2], vec![], vec![]));

        assert!(!ns.try_out_remove(1));
        assert!(ns.try_out_remove(2));

        assert_eq!(tripple(&ns), (vec![], vec![], vec![]));

        ns.out_add(1);
        ns.out_add(3);

        assert_eq!(tripple(&ns), (vec![1, 3], vec![], vec![]));

        ns.in_clear();

        assert_eq!(tripple(&ns), (vec![1, 3], vec![], vec![]));

        ns.out_clear();

        assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
    }

    #[test]
    fn in_only() {
        let mut ns = Neighborhood::new();
        ns.in_add(1);
        ns.in_add(2);

        assert_eq!(tripple(&ns), (vec![], vec![], vec![1, 2]));

        assert!(!ns.try_in_remove(3));
        assert!(ns.try_in_remove(1));

        assert_eq!(tripple(&ns), (vec![], vec![], vec![2]));

        assert!(!ns.try_in_remove(1));
        assert!(ns.try_in_remove(2));

        assert_eq!(tripple(&ns), (vec![], vec![], vec![]));

        ns.in_add(1);
        ns.in_add(3);

        assert_eq!(tripple(&ns), (vec![], vec![], vec![1, 3]));

        ns.out_clear();

        assert_eq!(tripple(&ns), (vec![], vec![], vec![1, 3]));

        ns.in_clear();

        assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
    }

    #[test]
    fn undir_only() {
        let mut ns = Neighborhood::new();

        ns.undir_add(1);
        ns.undir_add(2);

        assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![]));

        assert!(!ns.try_undir_remove(3));
        assert!(ns.try_undir_remove(1));

        assert_eq!(tripple(&ns), (vec![], vec![2], vec![]));

        assert!(!ns.try_undir_remove(1));
        assert!(ns.try_undir_remove(2));

        assert_eq!(tripple(&ns), (vec![], vec![], vec![]));

        ns.undir_add(1);
        ns.undir_add(3);

        assert_eq!(tripple(&ns), (vec![], vec![1, 3], vec![]));
    }

    #[test]
    fn dir_to_undir() {
        {
            // out first
            let mut ns = Neighborhood::with_capacity(2);
            ns.out_add(1);
            assert_eq!(tripple(&ns), (vec![1], vec![], vec![]));

            ns.in_add(1);
            assert_eq!(tripple(&ns), (vec![], vec![1], vec![]));

            ns.in_add(2);
            assert_eq!(tripple(&ns), (vec![], vec![1], vec![2]));

            ns.out_add(2);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![]));

            ns.in_clear();
            assert_eq!(tripple(&ns), (vec![1, 2], vec![], vec![]));

            ns.out_clear();
            assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
        }

        {
            // in first
            let mut ns = Neighborhood::new();
            ns.in_add(1);
            assert_eq!(tripple(&ns), (vec![], vec![], vec![1]));

            ns.out_add(1);
            assert_eq!(tripple(&ns), (vec![], vec![1], vec![]));

            ns.out_add(2);
            assert_eq!(tripple(&ns), (vec![2], vec![1], vec![]));

            ns.in_add(2);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![]));

            ns.out_clear();
            assert_eq!(tripple(&ns), (vec![], vec![], vec![1, 2]));

            ns.in_clear();
            assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
        }
    }

    #[test]
    fn remove_undir_to_dir() {
        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            assert_eq!(tripple(&ns), (vec![], vec![1], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![], vec![1]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            assert_eq!(tripple(&ns), (vec![], vec![1], vec![]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![1], vec![], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![1]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![1], vec![2], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.out_add(5);
            ns.undir_add(1);
            ns.undir_add(2);
            assert_eq!(tripple(&ns), (vec![5], vec![1, 2], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![1]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.out_add(5);
            ns.undir_add(1);
            ns.undir_add(2);
            assert_eq!(tripple(&ns), (vec![5], vec![1, 2], vec![]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![1, 5], vec![2], vec![]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            ns.in_add(6);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![6]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![1, 6]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![6]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            ns.in_add(6);
            assert_eq!(tripple(&ns), (vec![], vec![1, 2], vec![6]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![1], vec![2], vec![6]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![], vec![2], vec![6]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            ns.out_add(5);
            ns.in_add(6);
            assert_eq!(tripple(&ns), (vec![5], vec![1, 2], vec![6]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![1, 6]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![6]));
        }

        {
            let mut ns = Neighborhood::new();
            ns.undir_add(1);
            ns.undir_add(2);
            ns.in_add(6);
            ns.out_add(5);
            assert_eq!(tripple(&ns), (vec![5], vec![1, 2], vec![6]));
            assert!(ns.try_in_remove(1));
            assert_eq!(tripple(&ns), (vec![1, 5], vec![2], vec![6]));
            assert!(ns.try_out_remove(1));
            assert_eq!(tripple(&ns), (vec![5], vec![2], vec![6]));
        }
    }
}
