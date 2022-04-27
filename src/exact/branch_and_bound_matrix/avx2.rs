use super::*;
use std::arch::x86_64::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

macro_rules! helper_mult16 {
    ($s:expr, $n:expr, $m_in:ident, $m0:ident, $m1:ident) => {{
        let mask0 = _mm256_set1_epi16(1_i16 << $n);
        let active0 = _mm256_cmpeq_epi16(_mm256_and_si256($m_in, mask0), mask0);

        let mask1 = _mm256_set1_epi16(2_i16 << $n);
        let active1 = _mm256_cmpeq_epi16(_mm256_and_si256($m_in, mask1), mask1);

        let adj0 = _mm256_set1_epi16(_mm256_extract_epi16::<$n>($m_in) as i16);
        let adj1 = _mm256_set1_epi16(_mm256_extract_epi16::<{ $n + 1 }>($m_in) as i16);

        $m0 = _mm256_or_si256($m0, _mm256_and_si256(adj0, active0));
        $m1 = _mm256_or_si256($m1, _mm256_and_si256(adj1, active1));
    }};
}

#[target_feature(enable = "avx2")]
pub unsafe fn transitive_closure16(matrix: *mut u16, n: Node) {
    let mut matrix_reg = _mm256_load_si256(matrix as *const __m256i);

    for _ in 0..4 {
        let mut matrix_reg0 = matrix_reg;
        let mut matrix_reg1 = matrix_reg;

        helper_mult16!(n, 0, matrix_reg, matrix_reg0, matrix_reg1);
        helper_mult16!(n, 2, matrix_reg, matrix_reg0, matrix_reg1);
        helper_mult16!(n, 4, matrix_reg, matrix_reg0, matrix_reg1);
        helper_mult16!(n, 6, matrix_reg, matrix_reg0, matrix_reg1);
        helper_mult16!(n, 8, matrix_reg, matrix_reg0, matrix_reg1);
        helper_mult16!(n, 10, matrix_reg, matrix_reg0, matrix_reg1);
        if n >= 12 {
            helper_mult16!(n, 12, matrix_reg, matrix_reg0, matrix_reg1);
            helper_mult16!(n, 14, matrix_reg, matrix_reg0, matrix_reg1);
        }

        matrix_reg = _mm256_or_si256(matrix_reg0, matrix_reg1);
    }

    _mm256_store_si256(matrix as *mut __m256i, matrix_reg);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[target_feature(enable = "avx2")]
pub unsafe fn transitive_closure32(matrix: *mut u32, n: Node) {
    debug_assert_eq!(matrix.align_offset(32), 0);
    let mut mat = [
        _mm256_load_si256((matrix as *const __m256i).add(0)),
        _mm256_load_si256((matrix as *const __m256i).add(1)),
        _mm256_load_si256((matrix as *const __m256i).add(2)),
        _mm256_load_si256((matrix as *const __m256i).add(3)),
    ];

    let k = (n + 1) / 2;

    for _ in 0..5 {
        for i in 0..k {
            let adj0 = _mm256_set1_epi32(*matrix.add(2 * i as usize) as i32);
            let adj1 = _mm256_set1_epi32(*matrix.add(2 * i as usize + 1) as i32);

            let mask0 = _mm256_set1_epi32((1u32 << (2 * i)) as i32);
            let mask1 = _mm256_set1_epi32((2u32 << (2 * i)) as i32);

            for m in &mut mat {
                *m = _mm256_or_si256(
                    *m,
                    _mm256_or_si256(
                        _mm256_and_si256(
                            _mm256_cmpeq_epi32(_mm256_and_si256(*m, mask0), mask0),
                            adj0,
                        ),
                        _mm256_and_si256(
                            _mm256_cmpeq_epi32(_mm256_and_si256(*m, mask1), mask1),
                            adj1,
                        ),
                    ),
                );
            }
        }

        for (i, m) in mat.iter().enumerate() {
            _mm256_store_si256((matrix as *mut __m256i).add(i), *m);
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn transitive_closure64(matrix: *mut u64, n: Node) {
    debug_assert_eq!(matrix.align_offset(32), 0);
    let k = (n as usize + 3) / 4;

    for _ in 0..6 {
        for i in 0..n as usize {
            let adj = *matrix.add(i);
            if adj == 0 {
                continue;
            }
            let adj = _mm256_set1_epi64x(adj as i64);
            let mask = _mm256_set1_epi64x((1u64 << i) as i64);

            for j in 0..k {
                let row = _mm256_load_si256((matrix as *const __m256i).add(j));
                let new_row = _mm256_or_si256(
                    row,
                    _mm256_and_si256(adj, _mm256_cmpeq_epi64(_mm256_and_si256(row, mask), mask)),
                );
                _mm256_store_si256((matrix as *mut __m256i).add(j), new_row);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn transitive_closure128(matrix: *mut u128, n: Node) {
    // this implementation is not optimal in the sense that we only have 16 YMM*
    // registers in AVX2 and the matrix needs all of them.

    assert_eq!(matrix.align_offset(32), 0);

    let k = (n as usize + 1) / 2;

    unsafe fn load_and_mask(matrix: *mut u128, i: usize) -> (__m256i, __m256i) {
        let adj = {
            let adj = _mm_load_si128(matrix.add(i) as *const __m128i);
            _mm256_set_m128i(adj, adj)
        };

        let mask = _mm256_set1_epi64x((1u64 << (i % 64)) as i64);

        (adj, mask)
    }

    for _ in 0..7 {
        let mut converged = true;
        for i in 0..(n as usize + 3) / 4 {
            let (a0, m0) = load_and_mask(matrix, 4 * i);
            let (a1, m1) = load_and_mask(matrix, 4 * i + 1);
            let (a2, m2) = load_and_mask(matrix, 4 * i + 2);
            let (a3, m3) = load_and_mask(matrix, 4 * i + 3);

            for j in 0..k {
                let row = _mm256_load_si256((matrix as *const __m256i).add(j));

                let cmp = if 4 * i < 64 {
                    _mm256_unpacklo_epi64(row, row)
                } else {
                    _mm256_unpackhi_epi64(row, row)
                };

                let r0 = _mm256_and_si256(a0, _mm256_cmpeq_epi64(_mm256_and_si256(cmp, m0), m0));
                let r1 = _mm256_and_si256(a1, _mm256_cmpeq_epi64(_mm256_and_si256(cmp, m1), m1));
                let r2 = _mm256_and_si256(a2, _mm256_cmpeq_epi64(_mm256_and_si256(cmp, m2), m2));
                let r3 = _mm256_and_si256(a3, _mm256_cmpeq_epi64(_mm256_and_si256(cmp, m3), m3));

                let res = _mm256_or_si256(
                    row,
                    _mm256_or_si256(_mm256_or_si256(r0, r1), _mm256_or_si256(r2, r3)),
                );

                _mm256_store_si256((matrix as *mut __m256i).add(j), res);

                if converged {
                    let diff = _mm256_xor_si256(row, res); // all zeros if equal
                    converged &= _mm256_testz_si256(diff, diff) == 1; // testz returns 1 if diff is all zeros
                }
            }
        }
        if converged {
            break;
        }
    }
}
