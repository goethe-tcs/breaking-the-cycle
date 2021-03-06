use crate::graph::{GraphOrder, Node};
use core::slice;
use fxhash::{FxBuildHasher, FxHashMap};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use std::borrow::Borrow;
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::slice::Iter;

//TODO: Compare to IndexMap (https://github.com/bluss/indexmap)

/// Wrapper around a HashSet and a Vector to provide insertion, removal and indexing in O(1) at
/// the expense of memory
#[derive(Clone)]
pub struct HashSetVec<T> {
    map: FxHashMap<T, usize>,
    vec: Vec<T>,
}

impl<T> HashSetVec<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            vec: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default()),
            vec: Vec::with_capacity(capacity),
        }
    }

    /// Checks if a node is in the set in O(1)
    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.contains_key(value)
    }

    /// Inserts the element at the end in O(1)
    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }

        let i = self.vec.len();
        self.vec.push(value.clone());
        self.map.insert(value, i);
        true
    }

    /// Inserts the element at the specified index in O(n) on average. Elements with an index equal
    /// to or greater than the passed in `index` are shifted to the right.
    pub fn insert_at(&mut self, index: usize, value: T) {
        assert!(index <= self.len());

        let dirty_index = if let Some(old_index) = self.get_index(&value) {
            self.vec.remove(old_index);
            *self.map.get_mut(&value).unwrap() = index;
            self.vec.insert(index, value);
            old_index.min(index + 1)
        } else {
            self.vec.insert(index, value.clone());
            self.map.insert(value, index);
            index + 1
        };

        self.fix_successors(dirty_index);
    }

    /// Removes the element in O(1). The removed element is replaced by the last element.
    pub fn swap_remove<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        if let Some(&i) = self.map.get(value) {
            // remove element
            self.map.remove(value);
            let removed_value = self.vec.swap_remove(i);

            // update index of swapped element in hashmap
            if i < self.vec.len() {
                let new_value_at_i = &self.vec[i];
                self.map.insert(new_value_at_i.clone(), i);
            }
            Some(removed_value)
        } else {
            None
        }
    }

    /// Removes the element in O(n) on average. The elements after the deleted one are shifted to
    /// the left.
    pub fn shift_remove<Q: ?Sized>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        if let Some(&i) = self.map.get(value) {
            self.map.remove(value);
            let removed_value = self.vec.remove(i);
            self.fix_successors(i);
            Some(removed_value)
        } else {
            None
        }
    }

    /// Shift removes all of the passed in nodes and then performs cleanup logic afterwards. This is
    /// faster than multiple calls to [HashSetVec::shift_remove]
    pub fn shift_remove_bulk<'a>(&mut self, values: impl Iterator<Item = &'a T>)
    where
        T: Eq + Hash + Clone + 'static,
    {
        let min_dirty_index = values
            .into_iter()
            .filter_map(|value| self.map.get(value).map(|&index| (value, index)))
            .sorted_by_key(|&(_, index)| index)
            .rev()
            .map(|(value, index)| {
                self.map.remove(value);
                self.vec.remove(index);
                index
            })
            .min();

        if let Some(min_dirty_index) = min_dirty_index {
            self.fix_successors(min_dirty_index);
        }
    }

    /// Removes and returns the element with the highest index of the set in O(1)
    pub fn pop(&mut self) -> Option<T> {
        self.vec.pop().map(|deleted_vale| {
            self.map.remove(&deleted_vale);
            deleted_vale
        })
    }

    /// Returns the index of `element` in O(1)
    pub fn get_index(&self, element: &T) -> Option<usize> {
        self.map.get(element).copied()
    }

    /// Returns a reference to one random element, or None if the slice is empty.
    pub fn choose<R: Rng>(&self, rng: &mut R) -> Option<&T> {
        self.vec.choose(rng)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.vec
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.vec.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.vec.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.len() == 0
    }

    /// Fixes indices of all elements from `index` to `self.vec.len()`.
    fn fix_successors(&mut self, index: usize) {
        for new_i in index..self.vec.len() {
            let element = self.vec.index(new_i);
            if let Some(dirty_i) = self.map.get_mut(element) {
                *dirty_i = new_i;
            }
        }
    }
}

impl<T> Default for HashSetVec<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Extend<T> for HashSetVec<T>
where
    T: Eq + Hash + Clone,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            let i = self.vec.len();
            self.map.insert(value.clone(), i);
            self.vec.insert(i, value);
        }
    }
}

impl<T: Clone> IntoIterator for HashSetVec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl<'a, T: Clone> IntoIterator for &'a HashSetVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.vec.iter()
    }
}

impl<'a, T: Clone> IntoIterator for &'a mut HashSetVec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.vec.iter_mut()
    }
}

impl<T> From<HashSetVec<T>> for Vec<T> {
    fn from(value: HashSetVec<T>) -> Self {
        value.vec
    }
}

impl<T> FromIterator<T> for HashSetVec<T>
where
    T: Eq + Hash + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut res = Self::new();
        res.extend(iter);

        res
    }
}

impl HashSetVec<Node> {
    pub fn from_graph<G: GraphOrder>(graph: &G) -> Self {
        let mut res = Self::with_capacity(graph.len());
        res.extend(graph.vertices());
        res
    }
}

impl<T> Index<usize> for HashSetVec<T>
where
    T: Eq + Hash,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.vec.index(index)
    }
}

impl<T> IndexMut<usize> for HashSetVec<T>
where
    T: Eq + Hash,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.vec.index_mut(index)
    }
}

#[cfg(feature = "test-case")]
#[cfg(test)]
mod test_cases {
    use super::super::macros::set_tests;
    use super::*;

    fn factory(elements: Vec<Node>, size: usize) -> HashSetVec<Node> {
        let mut res = HashSetVec::with_capacity(size);
        res.extend(elements);
        res
    }

    set_tests!(HashSetVec<Node>, factory);
}
