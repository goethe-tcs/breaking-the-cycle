pub mod bit_manip;
pub mod bitintr;
pub mod int_iterator;
pub mod int_subsets;

pub use bit_manip::*;
pub use int_iterator::*;
pub use int_subsets::*;

use glob::glob;
use itertools::Itertools;
use std::path::PathBuf;

/// Retrieves all file paths of a collection of glob patterns
pub fn expand_globs<'a, I>(glob_patterns: I) -> Vec<PathBuf>
where
    I: Iterator<Item = &'a String>,
{
    glob_patterns
        .flat_map(|glob_pattern| {
            glob(glob_pattern)
                .unwrap_or_else(|_| panic!("Failed to read input {:?}", glob_pattern))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| panic!("Failed to read input {:?}", glob_pattern))
        })
        .collect_vec()
}
