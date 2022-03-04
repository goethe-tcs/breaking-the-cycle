extern crate core;

use dfvs::bitset::BitSet;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::{DefaultWriter, FileFormat, GraphRead};
use dfvs::graph::{Connectivity, GraphDigest, GraphOrder, InducedSubgraph};
use dfvs::log::build_pace_logger_for_verbosity;
use dfvs::pre_processor_reduction::{PreprocessorReduction, ReductionState};
use glob::glob;
use itertools::Itertools;
use log::{info, trace, warn, LevelFilter};
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "dfvs-preprocessing",
    about = "Applies reduction rules to passed in graphs."
)]
struct Opt {
    /// Output directory. Will store kernels in the same directory as the input graph if no
    /// output directory is specified
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Include graph files with '_kernel' in their name. This is disabled by default so that
    /// kernels aren't reduced again
    #[structopt(long)]
    include_kernels: bool,

    /// Create a file for each strongly connected component of each kernel in addition to the kernel
    /// graph file. Strongly connected components with only one node and no self loop are omitted.
    #[structopt(long)]
    export_sccs: bool,

    /// Set the minimal length of the hash (in characters) that is included in each strongly
    /// connected components file name. The length of the hash used is extended adaptively if file
    /// name collisions occur. Defaults to the full hash length if not set and equal SCCs are thus
    /// overwritten.
    #[structopt(long)]
    scc_hash_size: Option<usize>,

    /// Input graph files
    #[structopt(required = true)]
    input: Vec<String>,

    /// Number of threads
    #[structopt(short = "p", long)]
    num_threads: Option<usize>,
}

const KERNEL_SUFFIX: &str = "_kernel";

fn main() -> std::io::Result<()> {
    let opt: Opt = Opt::from_args();
    build_pace_logger_for_verbosity(LevelFilter::Warn, opt.verbose);

    let graphs = opt
        .input
        .iter()
        .flat_map(|glob_pattern| {
            glob(glob_pattern)
                .unwrap_or_else(|e| panic!("Failed to read input {}: {}", glob_pattern, e))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|e| panic!("Failed to read input {}: {}", glob_pattern, e))
        })
        .collect_vec(); //collect to avoid infinite iterator

    info!("Reducing {} graphs...", graphs.len());

    if let Some(num_threads) = opt.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to set the number of threads used by rayon: {}", e));
    }

    graphs
        .par_iter()
        .try_for_each(|graph_path| process_file(&opt, graph_path))
}

fn process_file(opt: &Opt, graph_path: &Path) -> std::io::Result<()> {
    let mut file_extension = graph_path.extension().unwrap().to_str().unwrap().to_owned();
    let file_name = graph_path.file_name().unwrap().to_str().unwrap();

    if !opt.include_kernels && file_name.contains(KERNEL_SUFFIX) {
        warn!(
            "Skipping graph file because it is a kernel: {:?}",
            graph_path
        );
        warn!("Disable this behaviour with '--include-kernels'");
        return Ok(());
    }
    trace!("Reading file {:?}", graph_path);
    let file_format = FileFormat::from_str(&file_extension)?;
    let graph = AdjArrayIn::try_read_graph(file_format, graph_path)?;

    trace!("Reducing graph {:?}", graph_path);
    let mut reduct_state = PreprocessorReduction::from(graph);
    reduct_state.apply_rules_exhaustively(false);
    let reduced_graph = reduct_state.graph().remove_disconnected_verts().0;

    let output_dir = opt
        .output
        .clone()
        .unwrap_or_else(|| graph_path.parent().unwrap().to_path_buf());
    create_dir_all(output_dir.clone())?;

    file_extension.insert(0, '.'); // 'metis' -> '.metis'
    let file_name_stripped = file_name.strip_suffix(&file_extension).unwrap();

    let kernel_name = format!(
        "{}{}{}",
        &file_name_stripped, KERNEL_SUFFIX, &file_extension
    );
    let kernel_path = output_dir.join(kernel_name);
    trace!("Writing kernel {:?}", kernel_path);
    let writer = DefaultWriter::from_path(Some(kernel_path), Some(file_format))?;
    writer.write(&reduced_graph)?;

    if opt.export_sccs {
        trace!("Splitting kernel into sccs...");
        let sccs = reduced_graph
            .strongly_connected_components_no_singletons()
            .into_iter()
            .map(|scc| {
                reduced_graph
                    .vertex_induced(&BitSet::new_all_unset_but(reduced_graph.len(), scc))
                    .0
            });

        for scc in sccs {
            let scc_path = get_unique_scc_path(&scc, &output_dir, opt.scc_hash_size, |hash| {
                format!(
                    "{}_n{}_m{}_{}{}{}",
                    &file_name_stripped,
                    scc.number_of_nodes(),
                    scc.number_of_edges(),
                    hash,
                    KERNEL_SUFFIX,
                    &file_extension
                )
            });

            trace!("Writing scc {:?}...", scc_path);
            let writer = DefaultWriter::from_path(Some(scc_path), Some(file_format))?;
            writer.write(&scc)?;
        }
    }

    Ok(())
}

fn get_unique_scc_path(
    scc: &(impl GraphOrder + GraphDigest),
    dir: impl AsRef<Path>,
    min_hash_size: Option<usize>,
    name_formatter: impl Fn(&str) -> String,
) -> PathBuf {
    let hash = scc.digest_sha256();
    let mut hash_len = min_hash_size.unwrap_or(hash.len());

    loop {
        let scc_name = name_formatter(&hash[0..hash_len]);
        let scc_path = dir.as_ref().join(&scc_name);
        hash_len += 1;

        if !scc_path.exists() {
            return scc_path;
        }

        // Overwrite if full hash is not enough to make file name unique
        if hash_len > hash.len() {
            warn!("Overwriting scc at {:?}!", scc_path);
            return scc_path;
        }
    }
}
