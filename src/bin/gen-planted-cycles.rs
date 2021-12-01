#![deny(warnings)]

use log::error;
use std::path::PathBuf;
use structopt::StructOpt;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use dfvs::graph::{io::DefaultWriter, AdjArray, Node};
use dfvs::random_models::planted_cycles::generate_planted_cycles;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "gen-planted-cycles",
    about = "Generates a random graph from the planted-cycles model"
)]
struct Opt {
    /// Output file. `stdout` if not specified.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Number of nodes to generate
    #[structopt(short, long)]
    nodes: Node,

    /// Average degree
    #[structopt(short = "d", long)]
    avg_deg: f64,

    /// Size of DFVS
    #[structopt(short = "k", long)]
    size_dfvs: Node,

    /// Number of cycles
    #[structopt(short = "c", long)]
    num_cycles: Node,

    /// Number of nodes in each cycle
    #[structopt(short = "l", long, default_value = "3")]
    cycle_len: Node,

    /// Seed value
    #[structopt(short, long)]
    seed: Option<u64>,
}

fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();

    if opt.avg_deg > opt.nodes as f64 / 2.0 {
        error!("The average degree must not exceed nodes / 2");
    }

    if opt.nodes < opt.num_cycles * opt.cycle_len + opt.size_dfvs {
        error!("Need nodes >= num_cycles * cycle_len + size_dfvs");
    }

    if opt.num_cycles < opt.size_dfvs {
        error!("Need  num_cycles >= size_dfvs")
    }

    let mut gen = match opt.seed {
        Some(s) => Pcg64Mcg::seed_from_u64(s),
        None => Pcg64Mcg::from_entropy(),
    };

    let writer = DefaultWriter::from_path(opt.output)?;

    let (graph, _): (AdjArray, _) = generate_planted_cycles(
        &mut gen,
        opt.nodes,
        opt.avg_deg,
        opt.size_dfvs,
        opt.num_cycles,
        opt.cycle_len,
    );

    writer.write(&graph)?;

    Ok(())
}
