use crate::graph::{GraphOrder, Node};
use core::slice;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::ops::{Index, IndexMut};
use std::slice::Iter;

/// Can be used to store nodes of a graph, combines set and vector like methods in one data
/// structure.
///
/// **Is initialized with a fixed size and all nodes have to be in the range [0..capacity)!**
///
/// But it should perform faster than a hash-set as a result.
#[derive(Clone)]
pub struct NodeIndexSet {
    size: Node,
    vec: Vec<Node>,
    index_lookup: Vec<Option<usize>>,
}

impl NodeIndexSet {
    /// Creates an empty instance that has capacity for `size` nodes
    pub fn new(size: usize) -> Self {
        Self {
            size: size as Node,
            vec: Vec::with_capacity(size),
            index_lookup: vec![None; size],
        }
    }

    /// Creates a new instance that contains all vertices of the passed in graph
    pub fn from_graph(graph: &impl GraphOrder) -> Self {
        let mut result = Self::new(graph.len());
        result.extend(graph.vertices());
        result
    }

    /// Checks if a node is in the set in O(1)
    pub fn contains(&self, node: &Node) -> bool {
        debug_assert!(node < &self.size);
        self.index_lookup.index(*node as usize).is_some()
    }

    /// Inserts the element at the end in O(1)
    pub fn insert(&mut self, node: Node) -> bool {
        debug_assert!(node < self.size);
        if self.contains(&node) {
            return false;
        }

        self.index_lookup[node as usize] = Some(self.vec.len());
        self.vec.push(node);
        true
    }

    /// Inserts the element at the specified index in O(n) on average. Elements with an index equal
    /// to or greater than the passed in `index` are shifted to the right.
    pub fn insert_at(&mut self, index: usize, node: Node) {
        assert!(index <= self.len());

        let dirty_index = if let Some(old_index) = self.get_index(&node) {
            self.vec.remove(old_index);
            self.vec.insert(index, node);
            old_index.min(index + 1)
        } else {
            self.vec.insert(index, node);
            index + 1
        };

        self.index_lookup[node as usize] = Some(index);
        self.fix_successors(dirty_index);
    }

    /// Removes the element in O(1). The removed element is replaced by the last element.
    pub fn swap_remove(&mut self, node: &Node) -> Option<Node> {
        debug_assert!(node < &self.size);

        self.index_lookup[*node as usize].map(|removed_node_i| {
            self.index_lookup[*node as usize] = None;
            let removed_node = self.vec.swap_remove(removed_node_i);

            // Update index of swapped node. There is no swapped node if this method was called for
            // the last element of the vector
            if removed_node_i < self.vec.len() {
                let swapped_node = self.vec[removed_node_i];
                self.index_lookup[swapped_node as usize] = Some(removed_node_i);
            }

            removed_node
        })
    }

    /// Removes the element in O(n) on average. The elements after the deleted one are shifted to
    /// the left.
    pub fn shift_remove(&mut self, node: &Node) -> Option<Node> {
        debug_assert!(node < &self.size);

        self.index_lookup[*node as usize].map(|i| {
            self.index_lookup[*node as usize] = None;
            let removed_node = self.vec.remove(i);
            self.fix_successors(i);

            removed_node
        })
    }

    /// Shift removes all of the passed in nodes and then performs cleanup logic afterwards. This is
    /// faster than multiple calls to [NodeIndexSet::shift_remove]
    pub fn shift_remove_bulk<'a>(&mut self, nodes: impl Iterator<Item = &'a Node>) {
        let min_dirty_index = nodes
            .into_iter()
            .filter_map(|&node| self.index_lookup[node as usize].map(|index| (node, index)))
            .sorted_by_key(|&(_, index)| index)
            .rev()
            .map(|(node, index)| {
                self.index_lookup[node as usize] = None;
                self.vec.remove(index);
                index
            })
            .min();

        if let Some(min_dirty_index) = min_dirty_index {
            self.fix_successors(min_dirty_index);
        }
    }

    /// Removes and returns the element with the highest index of the set in O(1)
    pub fn pop(&mut self) -> Option<Node> {
        self.vec.pop().map(|removed_value| {
            self.index_lookup[removed_value as usize] = None;
            removed_value
        })
    }

    /// Returns the index of `element` in O(1)
    pub fn get_index(&self, node: &Node) -> Option<usize> {
        self.index_lookup[*node as usize]
    }

    /// Returns a reference to one random element, or None if the slice is empty.
    pub fn choose<R: Rng>(&self, rng: &mut R) -> Option<&Node> {
        self.vec.choose(rng)
    }

    pub fn as_slice(&self) -> &[Node] {
        &self.vec
    }

    pub fn iter(&self) -> Iter<'_, Node> {
        self.vec.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Node> {
        self.vec.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.size as usize
    }

    /// Fixes indices of all elements from `index` to `self.vec.len()`.
    fn fix_successors(&mut self, index: usize) {
        for new_i in index..self.vec.len() {
            let node = self.vec.index(new_i);
            self.index_lookup[*node as usize] = Some(new_i);
        }
    }
}

impl Extend<Node> for NodeIndexSet {
    fn extend<I: IntoIterator<Item = Node>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl IntoIterator for NodeIndexSet {
    type Item = Node;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a> IntoIterator for &'a NodeIndexSet {
    type Item = &'a Node;
    type IntoIter = slice::Iter<'a, Node>;

    fn into_iter(self) -> slice::Iter<'a, Node> {
        self.vec.iter()
    }
}

impl<'a> IntoIterator for &'a mut NodeIndexSet {
    type Item = &'a mut Node;
    type IntoIter = slice::IterMut<'a, Node>;

    fn into_iter(self) -> slice::IterMut<'a, Node> {
        self.vec.iter_mut()
    }
}

impl From<NodeIndexSet> for Vec<Node> {
    fn from(value: NodeIndexSet) -> Self {
        value.vec
    }
}

impl Index<usize> for NodeIndexSet {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        self.vec.index(index)
    }
}

impl IndexMut<usize> for NodeIndexSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.vec.index_mut(index)
    }
}

#[cfg(feature = "test-case")]
#[cfg(test)]
mod test_cases {
    use super::super::macros::set_tests;
    use super::*;

    fn factory(elements: Vec<Node>, size: usize) -> NodeIndexSet {
        let mut res = NodeIndexSet::new(size);
        res.extend(elements);
        res
    }

    set_tests!(NodeIndexSet, factory);
}
