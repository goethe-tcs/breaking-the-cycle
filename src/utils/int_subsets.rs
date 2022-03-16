/// Iterator to enumerate all integers with a given number of bits set
pub struct IntSubsetsOfSize {
    value: u64,
    num_bits_set: u32,
    max_value: u64,
}

impl IntSubsetsOfSize {
    /// Creates an interator to enumerate over all integers with a fixed number of bits
    /// and a given number of bits set. Size must be less than 64. The numbers are returned
    /// in strictly increasing order.
    ///
    /// # Example
    /// ```
    /// use dfvs::utils::int_subsets::IntSubsetsOfSize;
    /// use itertools::Itertools;
    /// let mut iter = IntSubsetsOfSize::new(2, 4);
    /// assert_eq!(iter.collect_vec(),
    ///    vec![0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]);
    /// ```
    pub fn new(num_set: u32, num_bits: u32) -> Self {
        assert!(num_bits <= 63);
        assert!(num_set <= num_bits);

        Self {
            value: (1 << num_set) - 1,
            num_bits_set: num_set,
            max_value: 1 << num_bits,
        }
    }

    /// Returns the number of bits set
    pub fn num_bits_set(&self) -> u32 {
        self.num_bits_set
    }
}

/// An iterator wrapping [`IntSubsetsOfSize`] that increases the number of bits set each time,
/// the inner iterator ends.
pub struct AllIntSubsets {
    num_bits: u32,
    k: u32,
    max_k: u32,
    iter: IntSubsetsOfSize,
}

impl AllIntSubsets {
    /// Creates an iterator to enumerate over all integers with `num_bits` bits (<= 64) starting
    /// with the smallest integer with `k` bits set. It then enumerates all integers with `k` bits
    /// in increasing order, increases k, and repeats until `k` == `num_bits`.
    ///
    /// # Example
    /// ```
    /// use dfvs::utils::int_subsets::AllIntSubsets;
    /// use itertools::Itertools;
    /// let mut iter = AllIntSubsets::start_with_bits_set(1, 3);
    /// assert_eq!(iter.collect_vec(), vec![0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111]);
    ///
    pub fn start_with_bits_set(k: u32, num_bits: u32) -> Self {
        Self {
            num_bits,
            k,
            max_k: num_bits,
            iter: IntSubsetsOfSize::new(k, num_bits),
        }
    }

    /// Shorthand to [`AllIntSubsets::start_with_bits_set(0, num_bits)`]
    pub fn new(num_bits: u32) -> Self {
        Self::start_with_bits_set(0, num_bits)
    }

    /// Returns the number of bits set currently set. This number may increase as the iterator
    /// is advanced.
    pub fn num_bits_set(&self) -> u32 {
        self.iter.num_bits_set()
    }

    /// Sets an (inclusive) upper limit on the number of bits set. It is illegal to set a lower than
    /// [`AllIntSubsets::num_bits_set()`].
    pub fn set_max_bits_set(&mut self, n: u32) {
        assert!(n >= self.num_bits_set());
        assert!(n <= self.num_bits);
        self.max_k = n;
    }
}

impl Iterator for IntSubsetsOfSize {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value >= self.max_value {
            return None;
        }

        let result = Some(self.value);

        // Use Gosper's Hack to generate the next larger number with the same number of bits set
        self.value = {
            let ffs = self.value.trailing_zeros() + 1;
            let val = self.value | self.value.overflowing_sub(1).0;
            let minus_neg = 0u64.overflowing_sub(!val).0;
            val.overflowing_add(1).0
                | ((!val & minus_neg).overflowing_sub(1).0)
                    .overflowing_shr(ffs)
                    .0
        };

        result
    }
}

impl Iterator for AllIntSubsets {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.iter.next();

        if result.is_none() {
            self.k += 1;
            if self.k <= self.max_k {
                self.iter = IntSubsetsOfSize::new(self.k, self.num_bits);
                return self.next();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn subsets() {
        assert_eq!(IntSubsetsOfSize::new(0, 3).collect_vec(), vec![0]);
        assert_eq!(IntSubsetsOfSize::new(1, 3).collect_vec(), vec![1, 2, 4]);
        assert_eq!(IntSubsetsOfSize::new(2, 3).collect_vec(), vec![3, 5, 6]);
        assert_eq!(IntSubsetsOfSize::new(3, 3).collect_vec(), vec![7]);
    }

    #[test]
    fn all_subsets() {
        assert_eq!(AllIntSubsets::new(0).collect_vec(), vec![0]);
        assert_eq!(AllIntSubsets::new(1).collect_vec(), vec![0, 1]);
        assert_eq!(AllIntSubsets::new(2).collect_vec(), vec![0, 1, 2, 3]);
        assert_eq!(
            AllIntSubsets::new(3).collect_vec(),
            vec![0, 1, 2, 4, 3, 5, 6, 7]
        );
        assert_eq!(
            AllIntSubsets::new(4).collect_vec(),
            vec![
                0b0000, 0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0101, 0b0110, 0b1001, 0b1010,
                0b1100, 0b0111, 0b1011, 0b1101, 0b1110, 0b1111
            ]
        );
    }

    #[test]
    fn set_max_bits_set() {
        let mut iter = AllIntSubsets::new(3);
        iter.set_max_bits_set(2);
        assert_eq!(iter.collect_vec(), vec![0, 1, 2, 4, 3, 5, 6]);
    }
}
