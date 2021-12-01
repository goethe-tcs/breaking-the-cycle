#![deny(warnings)]

use dfvs::graph::io::{PaceRead, PaceWrite};
use dfvs::graph::AdjListMatrix;
use std::io::{stdin, stdout};

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> std::io::Result<()> {
    #[cfg(feature = "pace-logging")]
    dfvs::log::build_pace_logger();

    let stdin = stdin();
    let graph = AdjListMatrix::try_read_pace(stdin.lock())?;
    let stdout = stdout();
    graph.try_write_pace(stdout.lock())?;

    Ok(())
}
