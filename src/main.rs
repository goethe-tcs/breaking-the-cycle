use dfvs::graph::Graph;
use dfvs::io::{PaceReader, PaceWriter};
use std::convert::TryFrom;
use std::io::{stdin, stdout};

fn main() -> std::io::Result<()> {
    #[cfg(feature = "pace-logging")]
    dfvs::log::build_pace_logger();

    let stdin = stdin();
    let reader = PaceReader(stdin.lock());
    let graph = Graph::try_from(reader)?;
    let stdout = stdout();
    let writer = PaceWriter::new(&graph, stdout.lock());
    writer.output()?;
    Ok(())
}
