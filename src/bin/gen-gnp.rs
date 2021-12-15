#![deny(warnings)]
use log::error;
use std::path::PathBuf;
use structopt::StructOpt;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use dfvs::graph::io::FileFormat;
use dfvs::graph::{io::DefaultWriter, AdjArray, Node};
use dfvs::random_models::gnp::generate_gnp;

#[derive(Debug, StructOpt)]
#[structopt(name = "gen-gnp", about = "Samples a directed Gilbert G(n,p) graph")]
struct Opt {
    /// Output file. `stdout` if not specified.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    #[structopt(short, long)]
    format: Option<FileFormat>,

    /// Number of nodes to generate
    #[structopt(short, long)]
    nodes: Node,

    /// Edge probability
    #[structopt(short, long)]
    probability: f64,

    /// Seed value
    #[structopt(short, long)]
    seed: Option<u64>,
}

fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();

    if !(0.0..=1.0).contains(&opt.probability) {
        error!("Probability has to be in interval [0, 1]");
    }

    let mut gen = match opt.seed {
        Some(s) => Pcg64Mcg::seed_from_u64(s),
        None => Pcg64Mcg::from_entropy(),
    };

    let writer = DefaultWriter::from_path(opt.output, opt.format)?;

    let graph: AdjArray = generate_gnp(&mut gen, opt.nodes, opt.probability);

    writer.write(&graph)?;

    Ok(())
}
