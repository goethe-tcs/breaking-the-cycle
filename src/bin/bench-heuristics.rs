extern crate core;

use dfvs::bench::fvs_bench::FvsBench;
use dfvs::bench::io::bench_dir;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::FileFormat;
use dfvs::heuristics::greedy::{greedy_dfvs, MaxDegreeSelector};
use dfvs::heuristics::local_search::rand_topo_strategy::RandomTopoStrategy;
use dfvs::heuristics::local_search::sim_anneal::sim_anneal;
use dfvs::log::build_pace_logger_for_level;
use dfvs::random_models::gnp::generate_gnp;
use glob::glob;
use log::{info, LevelFilter};
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "bench-heuristics",
    about = "Runs a benchmark for various heuristic algorithms that calculate the \
directed feedback vertex set of a graph"
)]
struct Opt {
    /// Output file.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    /// How often a algorithm graph pair is run
    #[structopt(short, long, default_value = "1")]
    iterations: usize,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,

    #[structopt(subcommand)]
    mode: Mode,
}

#[derive(Debug, StructOpt)]
#[structopt()]
enum Mode {
    /// Generate input graphs
    Generate {
        /// Seed used for graph generation
        #[structopt(short, long, default_value = "1")]
        seed: u64,

        /// Node count for generated graphs
        #[structopt(short, long, use_delimiter = true, default_value = "1000,2000")]
        nodes: Vec<u32>,

        /// Edge probability for graph generation
        #[structopt(short, long, use_delimiter = true, default_value = "0.0015")]
        probabilities: Vec<f64>,
    },
    /// Read input graphs
    Read {
        /// Input graph files
        #[structopt(required = true)]
        input: Vec<String>,
    },
}

fn main() -> std::io::Result<()> {
    let opt: Opt = Opt::from_args();

    let log_level = match opt.verbose {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        3 => LevelFilter::Trace,
        _ => panic!("Invalid verbosity level '{}'", opt.verbose),
    };
    build_pace_logger_for_level(log_level);

    let output = opt.output.unwrap_or_else(|| {
        bench_dir()
            .expect("Failed to create output directory!")
            .join("bench-heuristics.csv")
    });

    // get graphs
    let mut bench = FvsBench::new();
    match opt.mode {
        Mode::Generate {
            seed,
            nodes,
            probabilities,
        } => {
            info!("Generating graphs with seed {}...", seed);
            let mut rng = Pcg64::seed_from_u64(seed);

            for n in &nodes {
                for p in &probabilities {
                    let label = format!("n_{}_p_{}", n, p);
                    bench.add_graph(label, generate_gnp(&mut rng, *n, *p));
                }
            }
        }
        Mode::Read { input } => {
            info!("Reading graphs...");
            let files = input.iter().flat_map(|glob_pattern| {
                glob(glob_pattern)
                    .unwrap_or_else(|_| panic!("Failed to read input {}", glob_pattern))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| panic!("Failed to read input {}", glob_pattern))
            });

            for file in files {
                bench.add_graph_file(FileFormat::Metis, file.clone())?;
            }
        }
    };

    bench.add_algo("greedy", |graph: AdjArrayIn, _| {
        greedy_dfvs::<MaxDegreeSelector<_>, _, _>(graph)
    });

    bench.add_algo("sim_anneal", |graph: AdjArrayIn, _| {
        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let mut move_strategy = RandomTopoStrategy::new(&mut strategy_rng, 7);
        sim_anneal(
            &graph,
            &mut move_strategy,
            20,
            20,
            1.0,
            0.9,
            &mut sim_anneal_rng,
        )
    });

    bench.strict(false).iterations(opt.iterations).run(output)?;
    Ok(())
}
