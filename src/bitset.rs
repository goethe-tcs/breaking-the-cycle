use bitvec::prelude::*;
use core::{fmt, iter, mem};
use num::{NumCast, ToPrimitive};
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign, Div, Index};

#[derive(Clone, Default)]
pub struct BitSet {
    cardinality: usize,
    bit_vec: BitVec,
}

impl Ord for BitSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bit_vec.cmp(&other.bit_vec)
    }
}

impl PartialOrd for BitSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bit_vec.partial_cmp(&other.bit_vec)
    }
}

impl Debug for BitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let values: Vec<_> = self.iter().map(|i| i.to_string()).collect();
        write!(
            f,
            "BitSet {{ cardinality: {}, bit_vec: [{}]}}",
            self.cardinality,
            values.join(", "),
        )
    }
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.cardinality == other.cardinality && self.bit_vec == other.bit_vec
    }
}
impl Eq for BitSet {}

impl Hash for BitSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bit_vec.hash(state)
    }
}

#[inline]
fn subset_helper(a: &[usize], b: &[usize]) -> bool {
    if a.len() > b.len() {
        !a.iter()
            .zip(b.iter().chain(iter::repeat(&0usize)))
            .any(|(a, b)| (*a | *b) != *b)
    } else {
        !a.iter()
            .chain(iter::repeat(&0usize))
            .zip(b.iter())
            .any(|(a, b)| (*a | *b) != *b)
    }
}

const fn block_size() -> usize {
    mem::size_of::<usize>() * 8
}

impl BitSet {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            cardinality: 0,
            bit_vec: bitvec![0; size],
        }
    }

    pub fn from_bitvec(bit_vec: BitVec) -> Self {
        let cardinality = bit_vec.iter().filter(|b| **b).count();
        Self {
            cardinality,
            bit_vec,
        }
    }

    pub fn from_slice<T: Div<Output = T> + ToPrimitive + AddAssign + Default + Copy + Display>(
        size: usize,
        slice: &[T],
    ) -> Self {
        let mut bit_vec: BitVec = bitvec![0; size];
        slice.iter().for_each(|i| {
            bit_vec.set(NumCast::from(*i).unwrap(), true);
        });
        let cardinality = slice.len();
        Self {
            cardinality,
            bit_vec,
        }
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.cardinality == 0
    }

    #[inline]
    pub fn full(&self) -> bool {
        self.cardinality == self.bit_vec.len()
    }

    #[inline]
    pub fn new_all_set(size: usize) -> Self {
        Self {
            cardinality: size,
            bit_vec: bitvec![1; size],
        }
    }

    #[inline]
    pub fn is_disjoint_with(&self, other: &BitSet) -> bool {
        !self.bit_vec
            .as_raw_slice()
            .iter()
            .zip(other.as_slice().iter())
            .any(|(x, y)| *x ^ *y != *x | *y)
    }

    #[inline]
    pub fn intersects_with(&self, other: &BitSet) -> bool {
        !self.is_disjoint_with(other)
    }

    #[inline]
    pub fn is_subset_of(&self, other: &BitSet) -> bool {
        self.cardinality <= other.cardinality
            && subset_helper(self.bit_vec.as_raw_slice(), other.as_slice())
    }

    #[inline]
    pub fn is_superset_of(&self, other: &BitSet) -> bool {
        other.is_subset_of(self)
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.bit_vec.as_raw_slice()
    }

    #[inline]
    pub fn as_bitslice(&self) -> &BitSlice {
        self.bit_vec.as_bitslice()
    }

    #[inline]
    pub fn as_bit_vec(&self) -> &BitVec {
        &self.bit_vec
    }

    #[inline]
    pub fn set_bit(&mut self, idx: usize) -> bool {
        if !*self.bit_vec.get(idx).unwrap() {
            self.bit_vec.set(idx, true);
            self.cardinality += 1;
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn unset_bit(&mut self, idx: usize) -> bool {
        if *self.bit_vec.get(idx).unwrap() {
            self.bit_vec.set(idx, false);
            self.cardinality -= 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_vec.is_empty()
    }

    #[inline]
    pub fn or(&mut self, other: &BitSet) {
        if other.len() > self.bit_vec.len() {
            self.bit_vec.resize(other.len(), false);
        }
        for (x, y) in self
            .bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x |= y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn resize(&mut self, size: usize) {
        let old_size = self.bit_vec.len();
        self.bit_vec.resize(size, false);
        if size < old_size {
            self.cardinality = self.bit_vec.count_ones();
        }
    }

    #[inline]
    pub fn and(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x &= y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn and_not(&mut self, other: &BitSet) {
        for (x, y) in self
            .bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .zip(other.as_slice().iter())
        {
            *x &= !y;
        }
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn not(&mut self) {
        self.bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .for_each(|x| *x = !*x);
        self.cardinality = self.bit_vec.count_ones();
    }

    #[inline]
    pub fn unset_all(&mut self) {
        self.bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .for_each(|x| *x = 0);
        self.cardinality = 0;
    }

    #[inline]
    pub fn set_all(&mut self) {
        self.bit_vec
            .as_mut_raw_slice()
            .iter_mut()
            .for_each(|x| *x = std::usize::MAX);
        self.cardinality = self.bit_vec.len();
    }

    #[inline]
    pub fn has_smaller(&mut self, other: &BitSet) -> Option<bool> {
        let self_idx = self.get_first_set()?;
        let other_idx = other.get_first_set()?;
        Some(self_idx < other_idx)
    }

    #[inline]
    pub fn get_first_set(&self) -> Option<usize> {
        if self.cardinality != 0 {
            return self.get_next_set(0);
        }
        None
    }

    #[inline]
    pub fn get_next_set(&self, idx: usize) -> Option<usize> {
        if idx >= self.bit_vec.len() {
            return None;
        }
        let mut block_idx = idx / block_size();
        let word_idx = idx % block_size();
        let mut block = self.bit_vec.as_raw_slice()[block_idx];
        let max = self.bit_vec.as_raw_slice().len();
        block &= usize::MAX << word_idx;
        while block == 0usize {
            block_idx += 1;
            if block_idx >= max {
                return None;
            }
            block = self.bit_vec.as_raw_slice()[block_idx];
        }
        let v = block_idx * block_size() + block.trailing_zeros() as usize;
        if v >= self.bit_vec.len() {
            None
        } else {
            Some(v)
        }
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<u32> {
        let mut tmp = Vec::with_capacity(self.cardinality);
        for (i, _) in self
            .bit_vec
            .as_bitslice()
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
        {
            tmp.push(i as u32);
        }
        tmp
    }

    #[inline]
    pub fn at(&self, idx: usize) -> bool {
        self.bit_vec[idx]
    }

    #[inline]
    pub fn iter(&self) -> BitSetIterator {
        BitSetIterator {
            iter: self.bit_vec.as_raw_slice().iter(),
            block: 0,
            idx: 0,
            size: self.bit_vec.len(),
        }
    }
}

pub struct BitSetIterator<'a> {
    iter: ::std::slice::Iter<'a, usize>,
    block: usize,
    idx: usize,
    size: usize,
}

impl<'a> Iterator for BitSetIterator<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.size {
            return None;
        }
        while self.block == 0 {
            self.block = if let Some(&i) = self.iter.next() {
                if i == 0 {
                    self.idx += block_size();
                    continue;
                } else {
                    self.idx = ((self.idx + block_size() - 1) / block_size()) * block_size();
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.block.trailing_zeros() as usize;
        self.block >>= offset;
        self.block >>= 1;
        self.idx += offset + 1;
        Some(self.idx - 1)
    }
}

impl Index<usize> for BitSet {
    type Output = bool;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.bit_vec.index(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::bitset::BitSet;

    #[test]
    fn iter() {
        let mut bs = BitSet::new(256);

        let a: Vec<usize> = (0..256).filter(|i| i % 2 == 0).collect();
        for i in &a {
            bs.set_bit(*i);
        }

        let b: Vec<usize> = bs.iter().collect();
        assert_eq!(a, b);
        let mut c = Vec::new();
        let mut v = bs.get_next_set(0);
        while v.is_some() {
            c.push(v.unwrap());
            v = bs.get_next_set(v.unwrap() + 1);
        }
        assert_eq!(a, c);
    }

    #[test]
    fn get_set() {
        let n = 257;
        let mut bs = BitSet::new(n);
        for i in 0..n {
            assert_eq!(false, bs[i]);
        }
        for i in 0..n {
            bs.set_bit(i);
            assert_eq!(true, bs[i]);
        }

        for i in 0..n {
            bs.unset_bit(i);
            assert_eq!(false, bs[i]);
        }
    }

    #[test]
    fn logic() {
        let n = 257;
        let mut bs1 = BitSet::new_all_set(n);

        for i in 0..n {
            assert_eq!(true, bs1[i]);
        }

        let mut bs2 = BitSet::new(n);
        for i in (0..n).filter(|i| i % 2 == 0) {
            bs2.set_bit(i);
            bs1.unset_bit(i);
        }

        let mut tmp = bs1.clone();
        tmp.and(&bs2);
        for i in 0..n {
            assert_eq!(false, tmp[i]);
        }

        let mut tmp = bs1.clone();
        tmp.or(&bs2);
        for i in 0..n {
            assert_eq!(true, tmp[i]);
        }

        let mut tmp = bs1.clone();
        tmp.and_not(&bs2);
        for i in (0..n).filter(|i| i % 2 == 0) {
            assert_eq!(false, tmp[i]);
        }
    }
}