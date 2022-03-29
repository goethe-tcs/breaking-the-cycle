extern crate core;

use dfvs::bench::fvs_bench::{FvsBench, InnerIterations, Iterations};
use dfvs::bench::io::bench_dir;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::FileFormat;
use dfvs::heuristics::greedy::{greedy_dfvs, MaxDegreeSelector};
use dfvs::heuristics::local_search::rand_topo_strategy::RandomTopoStrategy;
use dfvs::heuristics::local_search::sim_anneal::sim_anneal;
use dfvs::log::build_pace_logger_for_verbosity;
use dfvs::random_models::gnp::generate_gnp;
use dfvs::utils::expand_globs;
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

    /// How many threads to use for benchmarking, default is the amount of physical cores of the
    /// system
    #[structopt(short = "p", long)]
    num_threads: Option<usize>,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

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
    build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

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
            for file in expand_globs(input.iter()) {
                bench.add_graph_file(FileFormat::Metis, file.clone())?;
            }
        }
    };

    if let Some(num_threads) = opt.num_threads {
        bench.num_threads(num_threads);
    }

    bench.add_algo("greedy", |graph: AdjArrayIn, _, _, _| {
        greedy_dfvs::<MaxDegreeSelector<_>, _, _>(graph)
    });

    bench.add_algo("sim_anneal", |graph: AdjArrayIn, _, _, _| {
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

    bench
        .strict(false)
        .iterations(Iterations::new(opt.iterations, InnerIterations::Fixed(1)))
        .run(output)?;
    Ok(())
}
