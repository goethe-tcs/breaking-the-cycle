#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![feature(test)]
#![feature(unchecked_math)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(custom_test_frameworks)]
#![feature(stdsimd)]
#![feature(is_sorted)]

extern crate core;

pub mod algorithm;
pub mod bench;
pub mod bitset;
pub mod exact;
pub mod graph;
pub mod heuristics;
pub mod kernelization;
pub mod log;
pub mod random_models;
pub mod signal_handling;
pub mod utils;
