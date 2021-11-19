#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]

pub mod bitset;
pub mod graph;
pub mod pre_processor;
pub mod reduction_rule;

#[cfg(feature = "pace-logging")]
pub mod log;
