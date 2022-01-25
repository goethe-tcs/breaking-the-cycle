use crate::graph::{GraphOrder, Node};
use core::slice;
use fxhash::{FxBuildHasher, FxHashMap};
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

    pub fn contains<Q: ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.map.contains_key(value)
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }

        let i = self.vec.len();
        self.vec.push(value.clone());
        self.map.insert(value, i);

        true
    }

    pub fn remove<Q: ?Sized>(&mut self, value: &Q)
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        let i = self.map.get(value);
        if i.is_none() {
            return;
        }

        // remove element
        let i = *i.unwrap();
        self.map.remove(value);
        self.vec.swap_remove(i);

        // update index of swapped element in hashmap
        if i < self.vec.len() {
            let new_value_at_i = &self.vec[i];
            self.map.insert(new_value_at_i.clone(), i);
        }
    }

    pub fn choose<R: Rng>(&self, rng: &mut R) -> Option<&T> {
        self.vec.choose(rng)
    }

    pub fn cloned_into_vec(&self) -> Vec<T> {
        self.vec.clone()
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
