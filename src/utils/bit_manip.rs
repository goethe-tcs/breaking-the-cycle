use crate::utils::bitintr::{Pdep, Pext};
use num::PrimInt;

pub trait BitManip: Copy + PrimInt + num::One + num::Zero {
    /// Equivalent to the `pext` instruction
    fn bit_extract(self, mask: Self) -> Self;

    /// Equivalent to the `pdep` instruction
    fn bit_deposit(self, mask: Self) -> Self;

    /// Returns pow(2, `pos`), i.e. a bit pattern where the `pos`-th least significant bit is set
    fn ith_bit_set(pos: usize) -> Self {
        Self::one() << pos
    }

    /// Tests whether the `pos`-th least significant bit is set (among others)
    fn is_ith_bit_set(self, pos: usize) -> bool {
        !(self & Self::ith_bit_set(pos)).is_zero()
    }

    /// Returns either Self::zero() or Self::one() depending on the value of the `pos`-th least significant bit
    fn ith_bit(self, pos: usize) -> Self {
        (self >> pos) & Self::one()
    }

    /// Returns a bit pattern, where exactly the `n` least significant bits are set
    fn lowest_bits_set(n: usize) -> Self {
        let bits: usize = 8 * std::mem::size_of::<Self>();
        if n == bits {
            Self::max_value()
        } else {
            Self::ith_bit_set(n) - Self::one()
        }
    }

    /// Returns swaps the bits at positions `pos0` and `pos1`
    fn exchange_bits(&self, pos0: usize, pos1: usize) -> Self {
        if pos0 == pos1 {
            return *self;
        }
        let u = pos0.min(pos1);
        let v = pos0.max(pos1);
        let diff = v - u;

        let mut result = *self;

        result = result & !(Self::ith_bit_set(u) | Self::ith_bit_set(v));
        result = result
            | ((*self & Self::ith_bit_set(u)) << diff)
            | ((*self & Self::ith_bit_set(v)) >> diff);

        result
    }
}

macro_rules! impl_bit_manip {
    ($t:ty) => {
        impl BitManip for $t {
            fn bit_extract(self, mask: Self) -> Self {
                self.pext(mask)
            }

            fn bit_deposit(self, mask: Self) -> Self {
                self.pdep(mask)
            }
        }
    };
}

impl_bit_manip!(u8);
impl_bit_manip!(u16);
impl_bit_manip!(u32);
impl_bit_manip!(u64);

impl BitManip for u128 {
    fn bit_extract(self, mask: Self) -> Self {
        let slo = self as u64;
        let shi = (self >> 64) as u64;

        let mlo = mask as u64;
        let mhi = (mask >> 64) as u64;

        let elo = slo.bit_extract(mlo) as u128;
        let ehi = shi.bit_extract(mhi) as u128;

        elo | (ehi << mlo.count_ones())
    }

    fn bit_deposit(self, mask: Self) -> Self {
        let mlo = mask as u64;
        let mhi = (mask >> 64) as u64;

        let slo = self as u64;
        let shi = (self >> mlo.count_ones()) as u64;

        (slo.bit_deposit(mlo) as u128) | ((shi.bit_deposit(mhi) as u128) << 64)
    }
}

/// Implements the `pext` instruction in software and applies the same mask to a mut slice
///
/// # Example
/// ```
/// use dfvs::utils::bit_manip::multi_bit_extract;
/// let mut values = [0b0000, 0b0010, 0b1000, 0b1010];
/// multi_bit_extract(0b1_1010, &mut values);
/// assert_eq!(values[0], 0b0000);
/// assert_eq!(values[1], 0b0001);
/// assert_eq!(values[2], 0b0010);
/// assert_eq!(values[3], 0b0011);
/// ```
pub fn multi_bit_extract<T: PrimInt>(mut mask: T, elems: &mut [T]) {
    let bits: usize = 8 * std::mem::size_of::<T>();

    for x in &mut *elems {
        *x = *x & mask;
    }

    let mut mk = !mask << 1;

    let mut len = 1;
    while len < bits {
        let mut mp = mk ^ (mk << 1);
        let mut j = 2;
        // parallel prefix "sum" to compute the parity of each
        while j < bits {
            // hopefully the compiler will unroll this completely
            mp = mp ^ (mp << j);
            j *= 2;
        }

        mk = mk & !mp;
        let mv = mp & mask;

        // compress mask
        mask = mask ^ mv | (mv >> len);

        for x in &mut *elems {
            let tmp = *x & mv;
            *x = (*x ^ tmp) | (tmp >> len);
        }

        len *= 2;
    }
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use num::{One, Zero};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    use itertools::Itertools;
    use test::Bencher;

    macro_rules! impl_bitmanip_test {
        ($t:ty) => {
            paste::item! {
                #[test]
                fn [< ith_bit_set_$t >]() {
                    let ntotal = (std::mem::size_of::<$t>() * 8) as usize;
                    for i in 0..ntotal {
                        let x = $t::ith_bit_set(i);
                        assert_eq!(x.count_ones(), 1);
                        assert_eq!(x.trailing_zeros(), i as u32);
                    }
                }

                #[test]
                fn [< lowest_bits_set_$t >]() {
                    let ntotal = (std::mem::size_of::<$t>() * 8) as usize;
                    for i in 0..ntotal+1 {
                        let x = $t::lowest_bits_set(i);
                        assert_eq!(x.count_ones(), i as u32);
                        assert_eq!(x.trailing_ones(), i as u32);
                    }
                }

                #[test]
                fn [< exchange_bits_$t >]() {
                    let ntotal = (std::mem::size_of::<$t>() * 8) as usize;
                    for i in 0..ntotal {
                        let xi = $t::ith_bit_set(i);
                        for j in 0..ntotal {
                            assert_eq!(xi.exchange_bits(i, j), $t::ith_bit_set(j));
                        }
                    }
                }


                #[test]
                fn [< is_ith_bit_set_$t >]() {
                    let ntotal = (std::mem::size_of::<$t>() * 8) as usize;
                    for i in 0..ntotal {
                        let xi = $t::ith_bit_set(i);
                        for j in 0..ntotal {
                            assert_eq!(xi.is_ith_bit_set(j), i == j);
                        }
                    }
                }

                #[test]
                fn [< ith_bit_$t >]() {
                    let ntotal = (std::mem::size_of::<$t>() * 8) as usize;
                    for i in 0..ntotal {
                        let xi = $t::ith_bit_set(i);
                        for j in 0..ntotal {
                            assert_eq!(xi.ith_bit(j), (i == j) as $t);
                        }
                    }
                }
            }
        };
    }

    impl_bitmanip_test!(u8);
    impl_bitmanip_test!(u16);
    impl_bitmanip_test!(u32);
    impl_bitmanip_test!(u64);
    impl_bitmanip_test!(u128);

    fn mask_with_n_bits<T: PrimInt, R: Rng>(rng: &mut R, ones: usize) -> T {
        let ntotal = std::mem::size_of::<T>() * 8;
        assert!(ones <= ntotal);
        let mut mask = T::zero();
        while mask.count_ones() < ones as u32 {
            mask = mask | (T::one() << rng.gen_range(0..ntotal));
        }
        mask
    }

    macro_rules! impl_extract_test {
        ($t:ty, $n:ident) => {
            paste::item! {
                #[test]
                fn [< single_bit_$n >]() {
                    let mut mask = $t::one();
                    while mask > 0 {
                        {
                            let mut values = [mask, !mask];
                            multi_bit_extract(mask, &mut values);
                            assert_eq!(values[0], $t::one());
                            assert_eq!(values[1], $t::zero());
                        }

                        mask = mask.overflowing_mul(2).0;
                    }
                }

                #[test]
                fn [< dual_bits_$n >]() {
                    let mut mask0 = $t::one();
                    while mask0 > 0 {
                        let mut mask1 = mask0.overflowing_mul(2).0;

                        while mask1 > 0 {
                            let mut values = [
                                mask0, mask1, mask0 | mask1, !(mask0 | mask1)
                            ];
                            multi_bit_extract(mask1 | mask0, &mut values);
                            assert_eq!(values[0], $t::one());
                            assert_eq!(values[1], $t::one() << 1);
                            assert_eq!(values[2], ($t::one() << 1) | $t::one());
                            assert_eq!(values[3], $t::zero());
                            mask1 = mask1.overflowing_mul(2).0;
                        }

                        mask0 = mask0.overflowing_mul(2).0;
                    }
                }

                #[test]
                fn [< pext_crossref_$n >]() {
                    const N: usize = std::mem::size_of::<$t>();
                    let mut rng = Pcg64::seed_from_u64(0x1234u64 * N as u64);

                    for _ in 0..100 {
                        let ones = rng.gen_range(1..=N);
                        let mask : $t = mask_with_n_bits(&mut rng, ones);

                        let orgs = {
                            let mut values = [$t::zero(); 10];
                            for x in &mut values {
                                *x = rng.gen();
                            }
                            values
                        };

                        let mut results = orgs;
                        multi_bit_extract(mask, &mut results);

                        for (&org, &res) in orgs.iter().zip(results.iter()) {
                            assert_eq!(org.bit_extract(mask), res);
                        }
                    }
                }
            }
        };
    }

    impl_extract_test!(u8, u8);
    impl_extract_test!(u16, u16);
    impl_extract_test!(u32, u32);
    impl_extract_test!(u64, u64);
    impl_extract_test!(u128, u128);

    macro_rules! impl_deposit_test {
        ($t:ty, $n:ident) => {
            paste::item! {
                #[test]
                fn [< pdep_crossref_$n >]() {
                    const N: usize = std::mem::size_of::<$t>();
                    let mut rng = Pcg64::seed_from_u64(0x1234u64 * N as u64);

                    let mut random_array = || {
                        let mut arr = [false ; N];
                        for x in arr.iter_mut() {
                            *x = rng.gen();
                        }
                        arr
                    };

                    let array_to_int = |arr: &[bool]| {
                        let mut res = $t::zero();
                        for (i, _) in arr.iter().enumerate().filter(|(_, &x)| x) {
                            res |= $t::ith_bit_set(i);
                        }
                        res
                    };

                    for _ in 0..100 {
                        let value = random_array();
                        let mask = random_array();

                        let mut j = 0;
                        let mut res = [false ; N];

                        for (i, &m) in mask.iter().enumerate() {
                            if m {
                                res[i] = value[j];
                                j += 1;
                            }
                        }

                        assert_eq!(array_to_int(&value).bit_deposit(array_to_int(&mask)), array_to_int(&res));
                    }
                }
            }
        }
    }

    impl_deposit_test!(u8, u8);
    impl_deposit_test!(u16, u16);
    impl_deposit_test!(u32, u32);
    impl_deposit_test!(u64, u64);
    impl_deposit_test!(u128, u128);

    #[target_feature(enable = "bmi2")]
    unsafe fn kernel_extract_bmi2<T: BitManip, const N: usize>(mask: T, values: &mut [T; N]) {
        for x in values {
            *x = x.bit_extract(mask);
        }
    }

    fn kernel_extract_wo_bmi2<T: BitManip, const N: usize>(mask: T, values: &mut [T; N]) {
        for x in values {
            *x = x.bit_extract(mask);
        }
    }

    fn mask_vector<T: PrimInt, R: Rng>(rng: &mut R) -> Vec<T> {
        let ntotal = std::mem::size_of::<T>() * 8;
        (0..ntotal * 16)
            .map(|x| mask_with_n_bits(rng, x % ntotal))
            .collect_vec()
    }

    macro_rules! impl_bench {
        ($t:ty, $n:expr) => {
            paste::item! {
                #[bench]
                fn [< bench_extract_bmi2_$t _$n >](b: &mut Bencher) {
                    let mut rng = Pcg64::seed_from_u64(0x1234u64 * $n);
                    let masks : Vec<$t> = mask_vector(&mut rng);
                    let mut values = [0 as $t ; $n];
                    for x in &mut values {
                        *x = rng.gen();
                    }

                    b.iter(|| {
                        let mut tmp = values;
                        for &m in &masks {
                            unsafe{kernel_extract_bmi2(m, &mut tmp);}
                        }
                        tmp
                    });
                }

                #[bench]
                fn [< bench_extract_wo_bmi2_$t _$n >](b: &mut Bencher) {
                    let mut rng = Pcg64::seed_from_u64(0x1234u64 * $n);
                    let masks : Vec<$t> = mask_vector(&mut rng);
                    let mut values = [0 as $t ; $n];
                    for x in &mut values {
                        *x = rng.gen();
                    }

                    b.iter(|| {
                        let mut tmp = values;
                        for &m in &masks {
                            kernel_extract_wo_bmi2(m, &mut tmp);
                        }
                        tmp
                    });
                }

                #[bench]
                fn [< bench_extract_multi_$t _$n >](b: &mut Bencher) {
                    let mut rng = Pcg64::seed_from_u64(0x1234u64 * $n);
                    let masks : Vec<$t> = mask_vector(&mut rng);
                    let mut values = [0 as $t ; $n];
                    for x in &mut values {
                        *x = rng.gen();
                    }

                    b.iter(|| {
                        let mut tmp = values;
                        for &m in &masks {
                            multi_bit_extract(m, &mut tmp);
                        }
                        tmp
                    });
                }

            }
        };
    }

    impl_bench!(u8, 1);
    impl_bench!(u8, 8);

    impl_bench!(u16, 1);
    impl_bench!(u16, 8);
    impl_bench!(u16, 16);

    impl_bench!(u32, 1);
    impl_bench!(u32, 8);
    impl_bench!(u32, 16);
    impl_bench!(u32, 32);

    impl_bench!(u64, 1);
    impl_bench!(u64, 8);
    impl_bench!(u64, 16);
    impl_bench!(u64, 32);
    impl_bench!(u64, 64);

    impl_bench!(u128, 1);
    impl_bench!(u128, 8);
    impl_bench!(u128, 16);
    impl_bench!(u128, 32);
    impl_bench!(u128, 64);
    impl_bench!(u128, 128);
}
