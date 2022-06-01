extern crate core;

use dfvs::bench::fvs_bench::{FvsBench, InnerIterations, Iterations};
use dfvs::bench::io::bench_dir;
use dfvs::exact::branch_and_bound_matrix::bb_stats::BBStats;
use dfvs::exact::branch_and_bound_matrix::{
    branch_and_bound_matrix, branch_and_bound_matrix_stats,
};
use dfvs::graph::io::FileFormat;
use dfvs::graph::matrix::AdjMatrixIn;
use dfvs::graph::{GraphEdgeEditing, GraphOrder};
use dfvs::log::build_pace_logger_for_verbosity;
use dfvs::random_models::gnp::generate_gnp;
use dfvs::utils::expand_globs;
use log::{info, LevelFilter};
use rand::prelude::*;
use rand_pcg::Pcg64;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "bench-branch-and-bound",
    about = "Benchmarks the branch-and-bound implementation. Example: \
     cargo run --features=cli --release --package dfvs --bin bench-branch-and-bound -- \
     -v --output-suffix vectorize_matrix_mul --input data/kernels/*"
)]
struct Opt {
    /// Output file.
    #[structopt(short, long, default_value = "")]
    output_suffix: String,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Input graph files
    #[structopt(short, long, default_value = "")]
    input: Vec<String>,

    /// Seed used for graph generation
    #[structopt(short, long, default_value = "123")]
    seed: u64,

    /// Max node count for generated graphs
    #[structopt(short, long, use_delimiter = true, default_value = "40")]
    gnp_max_nodes: u32,

    /// Edge probability for graph generation
    #[structopt(short, long, use_delimiter = true, default_value = "0.1,0.3,0.7")]
    gnp_probabilities: Vec<f64>,

    /// Min seconds the iterations should take on each graph
    #[structopt(short, long, default_value = "1.0")]
    min_seconds: u64,

    /// Min iterations on each graph
    #[structopt(short, long, default_value = "5")]
    min_iterations: usize,

    /// How many threads to use for benchmarking, default is the amount of physical cores of the
    /// system
    #[structopt(short = "p", long)]
    num_threads: Option<usize>,
}

fn main() -> std::io::Result<()> {
    let opt: Opt = Opt::from_args();
    build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let output = bench_dir()
        .expect("Failed to create output directory!")
        .join("bench-branch-and-bound-".to_owned() + opt.output_suffix.as_str() + ".csv");

    let mut bench = FvsBench::<AdjMatrixIn>::new();

    info!("Reading graphs...");
    for file in expand_globs(opt.input.iter()) {
        bench.add_graph_file(FileFormat::Metis, file.clone())?;
    }

    info!("Generating graphs with seed {}...", opt.seed);
    let mut rng = Pcg64::seed_from_u64(opt.seed);

    for n in 1..=opt.gnp_max_nodes {
        for p in &opt.gnp_probabilities {
            let label = format!("n_{}_p_{}", n, p);
            let mut graph: AdjMatrixIn = generate_gnp(&mut rng, n, *p);

            for i in graph.vertices_range() {
                graph.try_remove_edge(i, i);
            }

            bench.add_graph(label, graph);
        }
    }

    if let Some(num_threads) = opt.num_threads {
        bench.num_threads(num_threads);
    }

    bench
        .iterations(Iterations::new(
            1,
            InnerIterations::Adaptive(
                std::time::Duration::new(opt.min_seconds, 0),
                opt.min_iterations,
            ),
        ))
        .split_into_sccs(false)
        .add_algo(
            "branch-and-bound",
            |graph, buffer, iteration, num_iterations| {
                if iteration == num_iterations - 1 {
                    let mut stats = BBStats::new();
                    let solution = branch_and_bound_matrix_stats(&graph, None, &mut stats).unwrap();

                    stats.write_to_buffer(buffer);

                    solution
                } else {
                    branch_and_bound_matrix(&graph, None).unwrap()
                }
            },
        )
        .run(output)
}
