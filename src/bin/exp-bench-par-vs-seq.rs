extern crate core;

use dfvs::algorithm::TerminatingIterativeAlgorithm;
use dfvs::bench::fvs_bench::{FvsBench, InnerIterations, Iterations};
use dfvs::bench::io::logs_dir;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::FileFormat;
use dfvs::heuristics::local_search::sim_anneal::SimAnneal;
use dfvs::heuristics::local_search::topo::rand_topo_strategy::RandomTopoStrategy;
use dfvs::heuristics::local_search::topo::topo_config::TopoConfig;
use dfvs::heuristics::local_search::topo::topo_local_search::TopoLocalSearch;
use dfvs::heuristics::local_search::topo::vec_topo_config::VecTopoConfig;
use dfvs::log::build_pace_logger_for_verbosity;
use dfvs::utils::expand_globs;
use log::{info, LevelFilter};
use rand::prelude::*;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "exp-fvs-bench")]
struct Opt {
    /// Output file.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    /// How often a algorithm graph pair is run
    #[structopt(short, long, default_value = "1")]
    iterations: usize,

    /// How many moves are evaluated per stage by the simulated annealing algorithm
    #[structopt(short, long, default_value = "140")]
    stage_evals: usize,

    /// How many threads to use for benchmarking
    #[structopt(short = "p", long)]
    num_threads: usize,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Input graph files
    #[structopt(required = true)]
    input: Vec<String>,
}

fn main() -> std::io::Result<()> {
    let opt: Opt = Opt::from_args();
    build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let output = opt.output.clone().unwrap_or_else(|| {
        logs_dir().unwrap().join(format!(
            "exp-bench-par-vs-seq_p_{}_evals_{}.csv",
            opt.num_threads, opt.stage_evals
        ))
    });

    let mut bench = FvsBench::new();

    bench.num_threads(opt.num_threads);

    info!("Reading graphs...");
    for file in expand_globs(opt.input.iter()) {
        bench.add_graph_file(FileFormat::Metis, file.clone())?;
    }

    let stage_evals = opt.stage_evals;
    bench.add_algo("sim_anneal", move |graph: AdjArrayIn, writer, _, _| {
        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            RandomTopoStrategy::new(&mut strategy_rng, 7),
        );
        let mut sim_anneal =
            SimAnneal::new(local_search, stage_evals, 20, 1.0, 0.9, &mut sim_anneal_rng);
        let fvs = sim_anneal.run_to_completion().unwrap();
        sim_anneal.write_metrics(writer);

        fvs
    });

    bench
        .strict(false)
        .iterations(Iterations::new(opt.iterations, InnerIterations::Fixed(1)))
        .run(output)?;

    Ok(())
}
