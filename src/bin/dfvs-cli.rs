#![deny(warnings)]

use dfvs::graph::*;
use itertools::Itertools;
use log::*;
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io::{stdin, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

use dfvs::bitset::BitSet;
use dfvs::graph::io::MetisRead;
use dfvs::kernelization::*;

type Graph = AdjArrayUndir;

use dfvs::algorithm::TerminatingIterativeAlgorithm;
use dfvs::exact::BranchAndBound;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "dfvs-cli",
    about = "Computes a directed feedback vertex set for a given input graph."
)]
struct Opt {
    /// Input file, using the graph format of the PACE 2022 challenge.
    /// `stdin` if not specified.
    #[structopt(short, long, parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file. `stdout` if not specified.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    /// Output format
    #[structopt(short = "f", long, default_value = "pace")]
    output_format: String,

    /// Solver. 'heuristic', 'exact'. Defaults to Exact.
    /// Any invalid input fails silently to 'exact'.
    #[structopt(short, long, default_value = "exact")]
    mode: String,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,
}

// Solver ////////////////////////////////////////////
#[derive(PartialEq, Debug)]
enum Solver {
    Heuristic,
    Exact,
}

impl Default for Solver {
    fn default() -> Self {
        Solver::Exact
    }
}

impl TryFrom<&str> for Solver {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "heuristic" => Ok(Solver::Heuristic),
            "exact" => Ok(Solver::Exact),
            _ => Err(format!("'{}' is an invalid Solver.", value)),
        }
    }
}

// Output Format ////////////////////////////////////////////
#[derive(PartialEq, Debug)]
enum OutputFormat {
    Pace,
    Dot,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Pace
    }
}

impl TryFrom<&str> for OutputFormat {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "pace" => Ok(OutputFormat::Pace),
            "dot" => Ok(OutputFormat::Dot),
            _ => Err(format!("'{}' is an invalid OutputFormat.", value)),
        }
    }
}

fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();
    //let opt = Opt::from_iter(std::iter::empty::<std::ffi::OsString>());
    dfvs::log::build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        info!("With AVX2 support");
    }

    let mode: Solver =
        Solver::try_from(opt.mode.as_str()).expect("Failed parsing 'mode' parameter: ");
    let output_format: OutputFormat = OutputFormat::try_from(opt.output_format.as_str())
        .expect("Failed parsing 'output_format' parameter: ");
    info!("Running in mode {:?}", mode);

    let graph: Graph = match &opt.input {
        Some(path) => {
            info!("Read file {:?}", path);
            let file = File::open(path)?;
            Graph::try_read_metis(BufReader::new(file))?
        }
        None => {
            let stdin = stdin();
            Graph::try_read_metis(stdin.lock())?
        }
    };

    info!(
        "Input graph with n={}, m={}",
        graph.number_of_nodes(),
        graph.number_of_edges()
    );

    let mut super_reducer = SuperReducer::with_settings(
        graph.clone(),
        vec![
            Rules::Rule1,
            Rules::Rule3,
            Rules::Rule4,
            Rules::RestartRules,
            Rules::PIE,
            Rules::DiClique,
            Rules::C4,
            Rules::DOMN,
            Rules::RestartRules,
            Rules::Unconfined,
            Rules::STOP,
            Rules::CompleteNode,
            Rules::Crown(5000),
            Rules::Rule5(5000),
        ],
        true,
    );

    let (solution, sccs) = super_reducer.reduce().unwrap();
    let mut solution = solution.clone();
    let mut sccs = sccs.clone();
    sccs.sort_by_key(|(g, _)| g.len());

    for (graph, mapper) in sccs {
        let start = Instant::now();

        let scc_solution = match mode {
            Solver::Exact => {
                let mut algo = BranchAndBound::with_paranoia(graph.clone());
                algo.run_to_completion()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect_vec()
            }
            _ => panic!("Solver not supported"),
        };

        solution.extend(scc_solution.iter().map(|&x| mapper.old_id_of(x).unwrap()));
        info!(
            "Processed SCC with n={:>5} and m={:>7}. Solution cardinality {:>5}. Elapsed: {:>8}ms",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            scc_solution.len(),
            start.elapsed().as_millis()
        );
    }

    solution.sort_unstable();

    info!("Solution has size {}", solution.len());

    // Verify solution
    {
        info!("Verify solution {}", solution.len());
        let mask = BitSet::new_all_set_but(graph.len(), solution.iter().copied());

        // no doubles
        assert_eq!(graph.len() - mask.cardinality(), solution.len());

        let sol = graph.vertex_induced(&mask).0;
        assert!(sol.is_acyclic());
    }

    // Output solution
    if let Some(path) = opt.output {
        let writer = OpenOptions::new().write(true).create(true).open(path)?;
        output_result(writer, output_format, &graph, &solution)?;
    } else {
        output_result(std::io::stdout().lock(), output_format, &graph, &solution)?;
    }

    info!("Done");

    Ok(())
}

fn output_result<W: Write>(
    mut writer: W,
    format: OutputFormat,
    graph: &Graph,
    solution: &[Node],
) -> std::io::Result<()> {
    match format {
        OutputFormat::Pace => {
            let result_str = solution.iter().map(|&x| format!("{}", x + 1)).join("\n");
            writeln!(writer, "{}", result_str)?;
        }

        OutputFormat::Dot => {
            graph.try_write_dot_with_solution(writer, solution.iter().copied())?;
        }
    }

    Ok(())
}
