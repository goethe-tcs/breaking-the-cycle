use crate::graph::{GraphOrder, Node};
use core::slice;
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
        //TODO: Check if 'graph.len()' is ok or if something like 'graph.vertices().max()' is required
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
    pub fn insert(&mut self, node: Node) {
        debug_assert!(node < self.size);
        debug_assert!(!self.contains(&node));

        self.index_lookup[node as usize] = Some(self.vec.len());
        self.vec.push(node);
    }

    /// Inserts the element at the specified index in O(n) on average. Elements with an index equal
    /// to or greater than the passed in `index` are shifted to the right.
    pub fn insert_at(&mut self, index: usize, node: Node) {
        debug_assert!(node < self.size);
        debug_assert!(!self.contains(&node));

        self.vec.insert(index, node);
        self.index_lookup[node as usize] = Some(index);
        self.fix_successors(index + 1);
    }

    /// Removes the element in O(1). The removed element is replaced by the last element.
    pub fn swap_remove(&mut self, node: &Node) -> Node {
        debug_assert!(node < &self.size);
        debug_assert!(self.contains(node));

        let removed_node_i = self.index_lookup[*node as usize].unwrap();
        self.index_lookup[*node as usize] = None;
        let removed_node = self.vec.swap_remove(removed_node_i);

        // Update index of swapped node. There is no swapped node if this method was called for
        // the last element of the vector
        if removed_node_i < self.vec.len() {
            let swapped_node = self.vec[removed_node_i];
            self.index_lookup[swapped_node as usize] = Some(removed_node_i);
        }

        removed_node
    }

    /// Removes the element in O(n) on average. The elements after the deleted one are shifted to
    /// the left.
    pub fn shift_remove(&mut self, node: &Node) -> Node {
        debug_assert!(node < &self.size);

        assert!(!self.contains(node));
        let i = self.index_lookup[*node as usize].unwrap();
        self.index_lookup[*node as usize] = None;
        let removed_node = self.vec.remove(i);
        self.fix_successors(i);

        removed_node
    }

    /// Shift removes all of the passed in nodes and then performs cleanup logic afterwards. This is
    /// faster than multiple calls to [NodeIndexSet::shift_remove]
    pub fn shift_remove_bulk<'a>(&mut self, nodes: impl Iterator<Item = &'a Node>) {
        let min_dirty_index = nodes
            .into_iter()
            .filter_map(|&node| {
                if let Some(i) = self.index_lookup[node as usize] {
                    self.index_lookup[node as usize] = None;
                    self.vec.remove(i);
                    Some(i)
                } else {
                    None
                }
            })
            .min();

        if let Some(min_dirty_index) = min_dirty_index {
            self.fix_successors(min_dirty_index);
        }
    }

    /// Removes and returns the element with the highest index of the set in O(1)
    pub fn pop(&mut self) -> Option<Node> {
        if self.is_empty() {
            None
        } else {
            let last_element = self[(self.len() - 1)];
            Some(self.swap_remove(&last_element))
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn set_from(elements: Vec<Node>) -> NodeIndexSet {
        let mut res = NodeIndexSet::new(10);
        res.extend(elements);
        res
    }

    fn assert_set_vec(actual: NodeIndexSet, expected: Vec<Node>) {
        assert_eq!(actual.vec, expected);

        for (i, element) in expected.into_iter().enumerate() {
            assert_eq!(actual.get_index(&element).unwrap(), i);
        }
    }

    #[test]
    fn test_push() {
        let mut set = set_from(vec![0, 1]);
        set.insert(2);

        assert_set_vec(set, vec![0, 1, 2]);
    }

    #[test]
    fn test_push_emtpy() {
        let mut set = set_from(vec![]);
        set.insert(2);

        assert_set_vec(set, vec![2]);
    }

    #[test]
    fn test_insert() {
        let mut set = set_from(vec![1, 2, 4, 5]);
        set.insert_at(2, 3);

        assert_set_vec(set, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_insert_almost_at_end() {
        let mut set = set_from(vec![1, 3]);
        set.insert_at(1, 2);

        assert_set_vec(set, vec![1, 2, 3]);
    }

    #[test]
    fn test_insert_at_end() {
        let mut set = set_from(vec![1, 2]);
        set.insert_at(2, 3);

        assert_set_vec(set, vec![1, 2, 3]);
    }

    #[test]
    fn test_insert_empty() {
        let mut set = set_from(vec![]);
        set.insert_at(0, 5);

        assert_set_vec(set, vec![5]);
    }
}

#[cfg(feature = "test-case")]
#[cfg(test)]
mod test_cases {
    use super::*;
    use test_case::test_case;

    #[test_case(&[0], 2 => (vec![Some(0), None], vec![]))]
    #[test_case(&[0, 1, 2], 1 => (vec![Some(2)], vec![0, 1]))]
    #[test_case(&[1, 0, 2], 4 => (vec![Some(2), Some(0), Some(1), None], vec![]))]
    fn test_pop(input: &[Node], pop_amount: usize) -> (Vec<Option<u32>>, Vec<u32>) {
        let mut index_set = NodeIndexSet::new(input.len());
        index_set.extend(input.into_iter().copied());

        let mut popped_nodes = vec![];
        for _ in 0..pop_amount {
            popped_nodes.push(index_set.pop());
        }

        (popped_nodes, index_set.as_slice().to_vec())
    }
}
