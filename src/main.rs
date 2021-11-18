#![deny(warnings)]

use dfvs::graph::AdjListMatrix;
use dfvs::graph::io::{PaceRead, PaceWrite};
use std::io::{stdin, stdout};

fn main() -> std::io::Result<()> {
    #[cfg(feature = "pace-logging")]
    dfvs::log::build_pace_logger();

    let stdin = stdin();
    let graph = AdjListMatrix::try_read_pace(stdin.lock())?;
    let stdout = stdout();
    graph.try_write_pace(stdout.lock())?;

    Ok(())
}
