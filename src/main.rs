use dfvs::io::PaceReader;
use petgraph::matrix_graph::DiMatrix;
use std::convert::TryFrom;
use std::io::stdin;

fn main() -> std::io::Result<()> {
    let stdin = stdin();
    let reader = PaceReader(stdin.lock());
    let graph = DiMatrix::try_from(reader)?;
    Ok(())
}