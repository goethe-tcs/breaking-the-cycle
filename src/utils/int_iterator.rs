use num::{PrimInt, Unsigned};

/// Functions to iterate over the zero/one bits in a primitive unsigned integer
pub trait IntegerIterators: PrimInt + Unsigned {
    /// Iterates over all bit indices that are one from least significant position to most significant
    ///
    /// # Example
    ///
    /// ```
    /// use dfvs::utils::int_iterator::IntegerIterators;
    /// use itertools::Itertools;
    /// assert_eq!(0b0000_0000_u32.iter_ones().collect_vec(), vec![]);
    /// assert_eq!(0b0000_0001_u32.iter_ones().collect_vec(), vec![0]);
    /// assert_eq!(0b0000_1010_u32.iter_ones().collect_vec(), vec![1, 3]);
    /// ```
    fn iter_ones(self) -> IntegerIterator<Self> {
        IntegerIterator::new(self)
    }

    /// Iterates over all bit indices that are zero from least significant position to most significant
    ///
    /// # Example
    ///
    /// ```
    /// use dfvs::utils::int_iterator::IntegerIterators;
    /// use itertools::Itertools;
    /// assert_eq!(0b1111_1111_u8.iter_zeros().collect_vec(), vec![]);
    /// assert_eq!(0b1111_1110_u8.iter_zeros().collect_vec(), vec![0]);
    /// assert_eq!(0b1011_1101_u8.iter_zeros().collect_vec(), vec![1, 6]);
    /// ```
    fn iter_zeros(self) -> IntegerIterator<Self> {
        IntegerIterator::new(!self)
    }
}

impl IntegerIterators for u8 {}
impl IntegerIterators for u16 {}
impl IntegerIterators for u32 {}
impl IntegerIterators for u64 {}
impl IntegerIterators for u128 {}

pub struct IntegerIterator<T> {
    x: T,
}

impl<T> IntegerIterator<T> {
    fn new(x: T) -> Self {
        Self { x }
    }
}

impl<T> Iterator for IntegerIterator<T>
where
    T: PrimInt + Unsigned,
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.x == T::zero() {
            None
        } else {
            let zeros = self.x.trailing_zeros();
            self.x = self.x ^ (T::one() << (zeros as usize));

            Some(zeros)
        }
    }
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use test::Bencher;

    fn test_small_ones<T: PrimInt + Unsigned + IntegerIterators>() {
        assert_eq!(T::from(0u8).unwrap().iter_ones().collect_vec(), vec![]);
        assert_eq!(T::from(1u8).unwrap().iter_ones().collect_vec(), vec![0]);
        assert_eq!(T::from(2u8).unwrap().iter_ones().collect_vec(), vec![1]);
        assert_eq!(T::from(3u8).unwrap().iter_ones().collect_vec(), vec![0, 1]);
        assert_eq!(T::from(4u8).unwrap().iter_ones().collect_vec(), vec![2]);
        assert_eq!(
            T::from(0xf0u8).unwrap().iter_ones().collect_vec(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            T::from(0xffu8).unwrap().iter_ones().collect_vec(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn ones_u8() {
        test_small_ones::<u8>();
    }

    #[test]
    fn ones_u16() {
        test_small_ones::<u16>();
        assert_eq!(0x8000_u16.iter_ones().collect_vec(), vec![15]);
    }

    #[test]
    fn ones_u32() {
        test_small_ones::<u32>();
        assert_eq!(0x8000_u32.iter_ones().collect_vec(), vec![15]);
        assert_eq!(0x8000_0000_u32.iter_ones().collect_vec(), vec![31]);
    }

    #[test]
    fn ones_u64() {
        test_small_ones::<u64>();
        assert_eq!(0x8000_u64.iter_ones().collect_vec(), vec![15]);
        assert_eq!(0x8000_0000_u64.iter_ones().collect_vec(), vec![31]);
        assert_eq!(0x8000_0000_0000_0000u64.iter_ones().collect_vec(), vec![63]);
    }

    fn test_small_zeros<T: PrimInt + Unsigned + IntegerIterators>() {
        assert_eq!((!T::from(0u8).unwrap()).iter_zeros().collect_vec(), vec![]);
        assert_eq!((!T::from(1u8).unwrap()).iter_zeros().collect_vec(), vec![0]);
        assert_eq!((!T::from(2u8).unwrap()).iter_zeros().collect_vec(), vec![1]);
        assert_eq!(
            (!T::from(3u8).unwrap()).iter_zeros().collect_vec(),
            vec![0, 1]
        );
        assert_eq!((!T::from(4u8).unwrap()).iter_zeros().collect_vec(), vec![2]);
        assert_eq!(
            (!T::from(0xf0u8).unwrap()).iter_zeros().collect_vec(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            (!T::from(0xffu8).unwrap()).iter_zeros().collect_vec(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn zeros_u8() {
        test_small_zeros::<u8>();
    }

    #[test]
    fn zeros_u16() {
        test_small_zeros::<u16>();
    }

    #[test]
    fn zeros_u32() {
        test_small_zeros::<u32>();
    }

    #[test]
    fn zeros_u64() {
        test_small_zeros::<u64>();
    }

    // We have two options to deal with already detected one; we either mask them out or we shift them out
    // The benchmarks below suggest that there are no significant performance differences.
    #[bench]
    fn shift_out(b: &mut Bencher) {
        let mut gen = Pcg64::seed_from_u64(1234);

        b.iter(|| {
            let mut num: u64 = gen.gen();

            let mut offset: u32 = 0;
            let mut sum: u32 = 0;
            while num != 0 {
                let one = num.trailing_zeros() + 1;
                offset += one;
                sum += offset;
                num >>= one;
            }
            unsafe { sum.unchecked_sub(1u32) }
        });
    }

    #[bench]
    fn mask_out(b: &mut Bencher) {
        let mut gen = Pcg64::seed_from_u64(1234);

        b.iter(|| {
            let mut num: u64 = gen.gen();
            let mut sum = 0;
            while num != 0 {
                let one = num.trailing_zeros();
                sum += one + 1;
                num ^= 1 << one;
            }
            sum
        });
    }
}
