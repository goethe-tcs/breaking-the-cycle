//#![deny(warnings)]

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
use dfvs::exact::branch_and_bound::branch_and_bound;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::MetisRead;
use dfvs::kernelization::{PreprocessorReduction, ReductionState};

type Graph = AdjArrayIn;

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
    dfvs::log::build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let mode: Solver =
        Solver::try_from(opt.mode.as_str()).expect("Failed parsing 'mode' parameter: ");
    let output_format: OutputFormat = OutputFormat::try_from(opt.output_format.as_str())
        .expect("Failed parsing 'output_format' parameter: ");
    info!("Running in mode {:?}", mode);

    let graph: Graph = match &opt.input {
        Some(path) => {
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

    let mut solution = match mode {
        Solver::Exact => process_graph(&opt, graph.clone())?,
        _ => panic!("Solver not supported"),
    };
    solution.sort_unstable();

    info!("Solution has size {}", solution.len());

    // Verify solution
    {
        info!("Verify solution {}", solution.len());
        let sol = graph
            .vertex_induced(&BitSet::new_all_set_but(graph.len(), solution.clone()))
            .0;
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

fn process_graph(_opt: &Opt, graph: Graph) -> std::io::Result<Vec<Node>> {
    let mut reduct_state = PreprocessorReduction::from(graph);
    reduct_state.apply_rules_exhaustively(true);
    let (reduced_graph, red_mapper) = reduct_state.graph().remove_disconnected_verts();

    let mut fvs = reduct_state.fvs().to_vec();

    trace!("Splitting kernel into sccs...");
    let sccs = reduced_graph
        .strongly_connected_components_no_singletons()
        .into_iter()
        .map(|scc| {
            reduced_graph.vertex_induced(&BitSet::new_all_unset_but(reduced_graph.len(), scc))
        });

    for (scc, mapper) in sccs {
        if scc.number_of_nodes() >= 64 {
            panic!(
                "Cannot process SCCs with n={} and m={}",
                scc.number_of_nodes(),
                scc.number_of_edges()
            );
        }

        let start = Instant::now();
        let solution = branch_and_bound(&scc, None).unwrap();
        let mapper = NodeMapper::compose(&red_mapper, &mapper);

        info!(
            "Processed SCC with n={:4}, m={:6} k={:4} in {:6}us",
            scc.number_of_nodes(),
            scc.number_of_edges(),
            solution.len(),
            start.elapsed().as_micros()
        );

        for x in solution {
            fvs.push(mapper.old_id_of(x).unwrap());
        }
    }

    Ok(fvs)
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
