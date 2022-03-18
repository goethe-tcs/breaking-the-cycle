use itertools::Itertools;
use log::{info, log_enabled, trace, warn, Level};
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::io::{self, sink};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::io::keyed_csv_writer::KeyedCsvWriter;
use super::io::keyed_writer::KeyedWriter;
use crate::bench::io::bench_writer::{BenchWriter, DualWriter};
use crate::bench::io::fvs_writer::{FvsFileWriter, FvsWriter};
use crate::bench::io::keyed_buffer::KeyedBuffer;
use crate::bench::io::other_io_error;
use crate::bitset::BitSet;
use crate::graph::io::{FileFormat, GraphRead};
use crate::graph::*;
use crate::pre_processor_reduction::{PreprocessorReduction, ReductionState};

/// Measures the computation time and solution size of multiple DFVS-algorithms against multiple
/// graphs. Each algorithm is run for each graph at least once.
///
/// # Examples
/// Basic usage:
/// ```
/// # use dfvs::heuristics::greedy::{greedy_dfvs, MaxDegreeSelector};
/// # use dfvs::graph::io::FileFormat;
/// # use rand::prelude::*;
/// # use rand_pcg::Pcg64;
/// # use dfvs::graph::adj_array::AdjArrayIn;
/// # use dfvs::random_models::gnp::generate_gnp;
/// use dfvs::bench::fvs_bench::FvsBench;
///
/// # #[cfg(feature = "tempfile")]
/// fn main() -> std::io::Result<()> {
///     use dfvs::bench::fvs_bench::{InnerIterations, Iterations};
/// let temp_dir = tempfile::TempDir::new().unwrap();
///     let output_path = temp_dir.path().join("test_bench.csv");
///
///     let n = 20;
///     let p = 0.01;
///     let seed = 1;
///     let mut rng = Pcg64::seed_from_u64(seed);
///
///     let mut bench = FvsBench::new();
///
///     bench.add_graph_file(FileFormat::Pace, "data/netrep/1/football.pace")?;
///     bench.add_graph("my_random_graph", generate_gnp::<AdjArrayIn, _>(&mut rng, n, p));
///
///     bench.add_algo("my_algo", |graph, _writer, _iteration, _num_iterations| {
///         greedy_dfvs::<MaxDegreeSelector<_>, _, _>(graph)
///     });
///
///     bench
///         .num_threads(1)
///         .iterations(Iterations::new(1, InnerIterations::Fixed(1)))
///         .strict(false)
///         .reduce_graphs(true)
///         .split_into_sccs(false)
///         .strict(false)
///         .run(output_path)?;
///
///     Ok(())
/// }
/// # #[cfg(not(feature = "tempfile"))]
/// # fn main() -> std::io::Result<()> { Ok(())} //TODO: remove this workaround
/// ```
pub struct FvsBench<G> {
    strict: bool,
    reduce_graphs: bool,
    split_into_sccs: bool,
    iterations: Iterations,
    num_threads: usize,

    /// Input graphs with a label
    graphs: Vec<LabeledGraph<G>>,

    /// DFVS algorithms with a label
    algos: Vec<LabeledFvsAlgo<G>>,

    /// Writes the result DFVS of all design points
    fvs_writer: Arc<Mutex<dyn FvsWriter + Send>>,
}

type LabeledGraph<G> = (String, Arc<G>);

type FvsAlgo<G> =
    Arc<dyn Fn(G, &mut KeyedBuffer, Iteration, NumIterations) -> Vec<Node> + Send + Sync>;
type LabeledFvsAlgo<G> = (String, FvsAlgo<G>);

type Iteration = usize;
type NumIterations = usize;
type MinSeconds = std::time::Duration;
type MinIterations = usize;

#[derive(Copy, Clone, Debug)]
pub enum InnerIterations {
    Fixed(NumIterations),
    Adaptive(MinSeconds, MinIterations),
}

#[derive(Copy, Clone, Debug)]
pub struct Iterations {
    design_point_iterations: usize,
    inner_iterations: InnerIterations,
}

impl Iterations {
    pub fn new(design_point_iterations: usize, inner_iterations: InnerIterations) -> Self {
        Self {
            design_point_iterations,
            inner_iterations,
        }
    }
}

impl Default for Iterations {
    fn default() -> Self {
        Self {
            design_point_iterations: 1,
            inner_iterations: InnerIterations::Fixed(1),
        }
    }
}

impl<G> FvsBench<G>
where
    G: GraphOrder
        + AdjacencyListIn
        + AdjacencyTest
        + GraphEdgeEditing
        + GraphNew
        + GraphRead
        + Debug
        + Clone
        + Send
        + Sync,
{
    pub fn new() -> Self {
        Self {
            strict: false,
            reduce_graphs: true,
            split_into_sccs: true,
            iterations: Iterations::default(),
            num_threads: num_cpus::get_physical(),

            graphs: vec![],
            algos: vec![],
            fvs_writer: Arc::new(Mutex::new(sink())),
        }
    }

    pub fn add_graph(&mut self, label: impl Display, graph: G) -> &mut Self {
        self.graphs.push((label.to_string(), Arc::new(graph)));
        self
    }

    /// Adds a graph file using its name as the graph label
    pub fn add_graph_file(
        &mut self,
        file_format: FileFormat,
        file_path: impl AsRef<Path>,
    ) -> io::Result<&mut Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let graph = G::try_read_graph(file_format, &file_path)?;

        let file_name = file_path
            .file_name()
            .ok_or_else(|| other_io_error("Failed to retrieve name of graph file!"))?
            .to_str()
            .ok_or_else(|| other_io_error("Failed to retrieve name of graph file!"))?;

        Ok(self.add_graph(file_name, graph))
    }

    pub fn add_algo<F>(&mut self, label: impl Display, algo: F) -> &mut Self
    where
        F: Fn(G, &mut KeyedBuffer, Iteration, NumIterations) -> Vec<Node> + Send + Sync + 'static,
    {
        self.algos.push((label.to_string(), Arc::new(algo)));
        self
    }

    /// Whether to apply reduction rules to the graphs before passing them on to the benched
    /// algorithms.
    ///
    /// Default: `true`
    pub fn reduce_graphs(&mut self, value: bool) -> &mut Self {
        self.reduce_graphs = value;
        self
    }

    /// Whether to split the input graphs into strongly connected components or not. SCC's of size
    /// 1 without a self loop are discarded.
    ///
    /// Default: `true`
    pub fn split_into_sccs(&mut self, value: bool) -> &mut Self {
        self.split_into_sccs = value;
        self
    }

    /// Sets whether to return an error if one of the DFVS-algorithms produces an incorrect DFVS.
    ///
    /// Default: `false`
    pub fn strict(&mut self, value: bool) -> &mut Self {
        self.strict = value;
        self
    }

    /// Sets how often each algorithm is run per graph.
    ///
    /// Default: `1`
    pub fn iterations(&mut self, iterations: Iterations) -> &mut Self {
        assert!(
            iterations.design_point_iterations > 0,
            "Iterations must be greater than zero!"
        );
        match iterations.inner_iterations {
            InnerIterations::Fixed(iters) => {
                assert!(iters > 0, "Iterations must be greater than zero!")
            }
            InnerIterations::Adaptive(min_seconds, min_iters) => {
                assert!(
                    min_seconds.as_secs_f64() > 0.0,
                    "Minimum number of seconds must be greater than zero!"
                );
                assert!(
                    min_iters > 0,
                    "Minimum number of iterations must be greater than zero!"
                );
            }
        }
        self.iterations = iterations;
        self
    }

    /// Sets the maximum amount of threads used. `0` will leave the decision up to [rayon].
    ///
    /// Default: [num_cpus::get_physical()]
    pub fn num_threads(&mut self, value: usize) -> &mut Self {
        self.num_threads = value;
        self
    }

    /// Sets a directory where the result DFVS of each design point will be stored in a separate
    /// file.
    ///
    /// Default: `None`
    pub fn fvs_output_dir(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.fvs_writer = Arc::new(Mutex::new(FvsFileWriter::new(path.as_ref().to_path_buf())));
        self
    }

    /// Sets a writer that is used to write the result DFVS of each design point
    ///
    /// Default: [std::io::Sink]
    pub fn fvs_writer<W: FvsWriter + Send + 'static>(&mut self, writer: W) -> &mut Self {
        self.fvs_writer = Arc::new(Mutex::new(writer));
        self
    }

    /// Runs the benchmark using the passed in BenchWriter
    pub fn run_with_writer<W: BenchWriter + Send>(&self, writer: W) -> io::Result<()> {
        if self.algos.is_empty() {
            return Err(other_io_error("No algorithms provided for benchmarking!"));
        }
        if self.graphs.is_empty() {
            return Err(other_io_error("No graphs provided for benchmarking!"));
        }

        //TODO: Reduce graphs before running design points

        // create design point vector
        let writer = Arc::new(Mutex::new(writer));
        let design_points = self
            .graphs
            .iter()
            .cartesian_product(self.algos.iter())
            .cartesian_product(0..self.iterations.design_point_iterations)
            .map(|(((graph_label, graph), (algo_label, algo)), _)| {
                (
                    graph_label.as_str(),
                    graph.clone(),
                    algo_label.as_str(),
                    algo.clone(),
                    writer.clone(),
                    self.fvs_writer.clone(),
                )
            })
            .collect_vec();
        let design_points_total = design_points.len();

        // try setting the number of threads used by rayon
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build_global()
            .unwrap_or_else(|e| warn!("Failed to set the number of threads used by rayon: {}", e));

        info!(
            "Running benchmark with {} design points using {} threads...",
            design_points_total,
            rayon::current_num_threads()
        );

        // copy fields of self to avoid moving self into threads
        let strict = self.strict;
        let reduce_graphs = self.reduce_graphs;
        let split_into_sccs = self.split_into_sccs;
        let iterations = self.iterations;

        // run benchmark
        let result = design_points
            .into_par_iter()
            .enumerate()
            .map::<_, Result<(), io::Error>>(
                |(i, (graph_label, input_graph, algo_label, algo, writer, fvs_writer))| {
                    let mut metric_buffer = KeyedBuffer::new();
                    trace!("[id {}] input graph: {:?}", i, input_graph);

                    // TODO: Pass reduction rules in as 'graph-pre-processing-step' instead of using
                    //  apply_rules param

                    // reduce graph
                    let mut reduct_state = PreprocessorReduction::from(G::clone(&input_graph));
                    if reduce_graphs {
                        reduct_state.apply_rules_exhaustively(false);
                        trace!("[id {}] reduced graph: {:?}", i, reduct_state.graph());
                    }
                    let (reduct_graph, reduct_mapper) =
                        reduct_state.graph().remove_disconnected_verts();
                    let mut total_fvs = reduct_state.fvs().to_vec();

                    // write graph related metrics
                    metric_buffer.write("id", i);
                    metric_buffer.write("graph", graph_label);
                    metric_buffer.write("n", input_graph.len());
                    metric_buffer.write("m", input_graph.number_of_edges());
                    metric_buffer.write("n_reduced", reduct_graph.len());
                    metric_buffer.write("m_reduced", reduct_graph.number_of_edges());

                    //split graph into strongly connected components
                    let sccs = if split_into_sccs {
                        reduct_graph
                            .strongly_connected_components_no_singletons()
                            .into_iter()
                            .map(|scc| {
                                reduct_graph.vertex_induced(&BitSet::new_all_unset_but(
                                    reduct_graph.len(),
                                    scc,
                                ))
                            })
                            .collect_vec()
                    } else {
                        vec![(reduct_graph, NodeMapper::with_capacity(0))]
                    };
                    metric_buffer.write("scc_amount", sccs.len());
                    if log_enabled!(Level::Trace) {
                        for (scc_i, scc) in sccs.iter().enumerate() {
                            trace!("[id {}] scc {}:{:#?}", i, scc_i, scc);
                        }
                    }

                    let inner_iterations =
                        Self::num_inner_iterations(iterations, sccs.clone(), algo.clone());

                    let repeated_sccs = if inner_iterations > 1 {
                        std::iter::repeat_with(|| sccs.clone())
                            .take(inner_iterations as usize)
                            .collect_vec()
                    } else {
                        vec![sccs]
                    };

                    let (elapsed_time, algo_results) = {
                        let start = Instant::now();
                        let mut solution = None;
                        for (i, sccs) in repeated_sccs.into_iter().enumerate() {
                            solution = Some(
                                sccs.into_iter()
                                    .map(|(scc, scc_mapper)| {
                                        (
                                            algo(scc, &mut metric_buffer, i, inner_iterations),
                                            scc_mapper,
                                        )
                                    })
                                    .collect_vec(),
                            ); // collect needed to eagerly evaluate for benchmarking purposes
                        }
                        (start.elapsed().as_secs_f64(), solution.unwrap())
                    };

                    // combine fvs of strongly connected components and remap node names
                    let algo_fvs = algo_results
                        .into_iter()
                        .flat_map(|(fvs, scc_mapper)| scc_mapper.get_old_ids(fvs.into_iter()))
                        .map(|node| reduct_mapper.old_id_of(node).unwrap_or(node))
                        .collect_vec();

                    // add algo fvs to the reduction fvs
                    total_fvs.extend(algo_fvs.iter().copied());

                    // write algorithm related metrics
                    metric_buffer.write("algo", algo_label);
                    metric_buffer.write("fvs_reduction", reduct_state.fvs().len());
                    metric_buffer.write("fvs_algo", algo_fvs.len());
                    metric_buffer.write("fvs_total", total_fvs.len());
                    metric_buffer.write("inner_iterations", inner_iterations);
                    metric_buffer.write("elapsed_sec_algo", elapsed_time / inner_iterations as f64);

                    // apply fvs to graph (to print resulting graph & check if graph is acyclic)
                    let mut result_graph = G::clone(&input_graph);
                    result_graph.remove_edges_of_nodes(total_fvs.iter());
                    trace!("[id {}] fvs_reduction: {:?}", i, reduct_state.fvs());
                    trace!("[id {}] fvs_algo: {:?}", i, algo_fvs);
                    trace!("[id {}] fvs_total: {:?}", i, total_fvs);
                    trace!("[id {}] resulting_graph: {:?}", i, result_graph);

                    // check if result is acyclic
                    let is_acyclic = result_graph.is_acyclic();
                    metric_buffer.write("is_result_acyclic", is_acyclic);
                    if strict && !is_acyclic {
                        return Err(other_io_error(format!(
                            "Result of design point {} (algorithm '{}') was not acyclic!",
                            i, algo_label
                        )));
                    }

                    // write metrics
                    match writer.lock() {
                        Ok(mut writer) => {
                            writer.add_buffer_content(&mut metric_buffer)?;
                            writer.end_design_point()?;
                        }
                        Err(error) => return Err(other_io_error(error.to_string())),
                    }

                    // write fvs
                    match fvs_writer.lock() {
                        Ok(mut fvs_writer) => fvs_writer.write(graph_label, &total_fvs)?,
                        Err(error) => return Err(other_io_error(error.to_string())),
                    }

                    Ok(())
                },
            )
            .collect();

        match writer.lock() {
            Ok(mut writer) => writer.end_bench()?,
            Err(error) => return Err(other_io_error(error.to_string())),
        }

        result
    }

    /// Runs the benchmark using a DualWriter that writes bench results to standard output as well
    /// as to a csv file
    pub fn run(&self, csv_path: impl AsRef<Path>) -> io::Result<()> {
        let writer = DualWriter::new(
            KeyedWriter::new_std_out_writer(),
            KeyedCsvWriter::new(csv_path.as_ref().to_path_buf())?,
        );
        self.run_with_writer(writer)
    }

    fn num_inner_iterations(
        iterations: Iterations,
        sccs: Vec<(G, NodeMapper)>,
        algo: FvsAlgo<G>,
    ) -> usize {
        match iterations.inner_iterations {
            InnerIterations::Fixed(iters) => iters,
            InnerIterations::Adaptive(min_seconds, min_iters) => {
                let mut buf = KeyedBuffer::new();

                let start = Instant::now();
                sccs.into_iter().for_each(|(scc, _)| {
                    algo(scc, &mut buf, 0, 0);
                });
                let iterations_per_min_seconds =
                    (min_seconds.as_secs_f64() / start.elapsed().as_secs_f64()).round() as usize;
                iterations_per_min_seconds.max(min_iters)
            }
        }
    }
}

impl<G> Default for FvsBench<G>
where
    G: GraphOrder
        + AdjacencyListIn
        + AdjacencyTest
        + GraphEdgeEditing
        + GraphNew
        + GraphRead
        + Debug
        + Clone
        + Send
        + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests_bench {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use std::io::{sink, ErrorKind};

    #[test]
    #[should_panic]
    fn test_zero_iterations_panic() {
        FvsBench::new()
            .iterations(Iterations::new(0, InnerIterations::Fixed(0)))
            .add_graph("self_loop", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |_, _, _, _| vec![])
            .run_with_writer(sink())
            .unwrap();
    }

    #[test]
    fn test_strict_no_error() {
        FvsBench::default()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .add_algo("test_algo", |_, _, _, _| vec![0])
            .strict(true)
            .reduce_graphs(false)
            .run_with_writer(sink())
            .unwrap();
    }

    #[test]
    fn test_strict_error() {
        let error_kind = FvsBench::default()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .add_algo("test_algo", |_, _, _, _| vec![])
            .strict(true)
            .reduce_graphs(false)
            .run_with_writer(sink())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::Other);
    }

    #[test]
    fn test_zero_graphs_error() {
        let error_kind = FvsBench::new()
            .add_algo("test_algo", |_graph: AdjArrayIn, _, _, _| vec![])
            .run_with_writer(sink())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::Other);
    }

    #[test]
    fn test_zero_algos_error() {
        let error_kind = FvsBench::new()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .run_with_writer(sink())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::Other);
    }
}

#[cfg(feature = "tempfile")]
#[cfg(test)]
mod tests_bench_with_tempfile {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::io::PaceWrite;
    use csv::Reader;
    use std::fs::File;
    use std::io::Read;
    use std::ops::Index;

    fn read_bench_output(file_path: impl AsRef<Path>) -> Vec<Vec<String>> {
        let mut reader = Reader::from_path(file_path).unwrap();

        let headers = reader
            .headers()
            .unwrap()
            .iter()
            .map(String::from)
            .collect_vec();

        let records = reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .map(|record| {
                let mut vec = record.iter().map(String::from).collect_vec();
                vec[12] = "".to_string(); // remove unstable elapsed time entry
                vec
            })
            .sorted_by(|a, b| a.index(0).cmp(b.index(0)));

        let mut result = vec![headers];
        result.extend(records);
        result
    }

    fn get_bench_column_headers<'a>() -> Vec<&'a str> {
        vec![
            "id",
            "graph",
            "n",
            "m",
            "n_reduced",
            "m_reduced",
            "scc_amount",
            "algo",
            "fvs_reduction",
            "fvs_algo",
            "fvs_total",
            "inner_iterations",
            "elapsed_sec_algo",
            "is_result_acyclic",
        ]
    }

    #[test]
    fn test_reduce_graphs() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_reduce_graphs.csv");

        FvsBench::new()
            .strict(false)
            .add_graph(
                "test_graph",
                AdjArrayIn::from(&vec![
                    (0, 0),
                    (1, 2),
                    (1, 3),
                    (2, 1),
                    (2, 3),
                    (3, 1),
                    (3, 2),
                ]),
            )
            .add_algo("test_algo", |_, _, _, _| vec![])
            .reduce_graphs(true)
            .split_into_sccs(false)
            .run(output_path.clone())
            .unwrap();

        let actual_output = read_bench_output(output_path);
        let expected_output = vec![
            get_bench_column_headers(),
            vec![
                "0",
                "test_graph",
                "4",
                "7",
                "0",
                "0",
                "1",
                "test_algo",
                "3",
                "0",
                "3",
                "1",
                "",
                "true",
            ],
        ];
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_multiple_graphs() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_graph_provider.csv");

        let graph_path = temp_dir.path().join("test_graph_provider_graph");
        let graph_file = File::create(graph_path.clone()).unwrap();
        AdjArrayIn::from(&[(0, 0), (0, 1)])
            .try_write_pace(graph_file)
            .unwrap();

        FvsBench::new()
            .add_graph_file(FileFormat::Pace, graph_path)
            .unwrap()
            .add_graph(
                "self_loops",
                AdjArrayIn::from(&vec![(0, 0), (1, 1), (2, 2), (3, 3)]),
            )
            .add_graph("tiny_graph", AdjArrayIn::from(&vec![(0, 0), (1, 2)]))
            .add_algo("do_nothing", |_graph: AdjArrayIn, _, _, _| vec![])
            .strict(false)
            .reduce_graphs(false)
            .run(output_path.clone())
            .unwrap();

        let actual_output = read_bench_output(output_path);

        let expected_output = vec![
            get_bench_column_headers(),
            vec![
                "0",
                "test_graph_provider_graph",
                "2",
                "2",
                "2",
                "2",
                "1",
                "do_nothing",
                "0",
                "0",
                "0",
                "1",
                "",
                "false",
            ],
            vec![
                "1",
                "self_loops",
                "4",
                "4",
                "4",
                "4",
                "4",
                "do_nothing",
                "0",
                "0",
                "0",
                "1",
                "",
                "false",
            ],
            vec![
                "2",
                "tiny_graph",
                "3",
                "2",
                "3",
                "2",
                "1",
                "do_nothing",
                "0",
                "0",
                "0",
                "1",
                "",
                "false",
            ],
        ];
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_bench_design_point_count() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_bench_design_point_count.csv");

        let graph_path = temp_dir.path().join("test_bench_design_point_count_graph");
        let graph_file = File::create(graph_path.clone()).unwrap();
        AdjArrayIn::from(&[(0, 0), (0, 1)])
            .try_write_pace(graph_file)
            .unwrap();

        FvsBench::new()
            .add_graph_file(FileFormat::Pace, graph_path)
            .unwrap()
            .add_graph("5_n", AdjArrayIn::new(5))
            .add_graph("self_loop", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |_, _, _, _| vec![])
            .add_algo("test_algo", |_, _, _, _| vec![0])
            .strict(false)
            .iterations(Iterations::new(3, InnerIterations::Fixed(1)))
            .run(output_path.clone())
            .unwrap();

        let output = read_bench_output(output_path);
        assert_eq!(output.len(), 19);
    }

    #[test]
    fn test_bench_inner_iterations_count() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_bench_design_point_count.csv");

        let graph_path = temp_dir.path().join("test_bench_design_point_count_graph");
        let graph_file = File::create(graph_path.clone()).unwrap();
        AdjArrayIn::from(&[(0, 0), (0, 1)])
            .try_write_pace(graph_file)
            .unwrap();

        FvsBench::new()
            .add_graph_file(FileFormat::Pace, graph_path)
            .unwrap()
            .add_graph("5_n", AdjArrayIn::new(5))
            .add_graph("self_loop", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |g, buffer, iteration, total_iterations| {
                assert_eq!(total_iterations, 42);
                assert!(iteration < total_iterations);
                buffer.write(
                    format!("iteration_{}_of_{}", iteration, total_iterations),
                    "",
                );
                (0..g.number_of_nodes()).collect_vec()
            })
            .strict(false)
            .iterations(Iterations::new(3, InnerIterations::Fixed(42)))
            .split_into_sccs(false)
            .run(output_path.clone())
            .unwrap();

        let output = read_bench_output(output_path);
        assert_eq!(output.len(), 10);
        for i in 0..10 {
            assert_eq!(output[i].len(), get_bench_column_headers().len() + 42);
        }
    }

    #[test]
    fn test_bench_adaptive_iterations_min_iters() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_bench_design_point_count.csv");

        let graph_path = temp_dir.path().join("test_bench_design_point_count_graph");
        let graph_file = File::create(graph_path.clone()).unwrap();
        AdjArrayIn::from(&[(0, 0), (0, 1)])
            .try_write_pace(graph_file)
            .unwrap();

        FvsBench::new()
            .add_graph_file(FileFormat::Pace, graph_path)
            .unwrap()
            .add_algo(
                "do_nothing",
                |g: AdjArrayIn, buffer, iteration, total_iterations| {
                    std::thread::sleep(std::time::Duration::new(0, 10_000_000));
                    buffer.write(
                        format!("iteration_{}_of_{}", iteration, total_iterations),
                        "",
                    );
                    (0..g.number_of_nodes()).collect_vec()
                },
            )
            .strict(false)
            .iterations(Iterations::new(
                3,
                InnerIterations::Adaptive(std::time::Duration::new(0, 1_000_000), 3),
            ))
            .split_into_sccs(false)
            .run(output_path.clone())
            .unwrap();

        let output = read_bench_output(output_path);
        assert_eq!(output.len(), 4);
        for i in 0..4 {
            // Algo takes ca. 10 ms, min iterations is 3, min seconds is 1 ms, so number of
            // iterations should be 3 from the adaptive iterations calculation.
            assert_eq!(output[i].len(), get_bench_column_headers().len() + 3);
        }
    }

    #[test]
    fn test_bench_adaptive_iterations_min_seconds() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_bench_design_point_count.csv");

        let graph_path = temp_dir.path().join("test_bench_design_point_count_graph");
        let graph_file = File::create(graph_path.clone()).unwrap();
        AdjArrayIn::from(&[(0, 0), (0, 1)])
            .try_write_pace(graph_file)
            .unwrap();

        FvsBench::new()
            .add_graph_file(FileFormat::Pace, graph_path)
            .unwrap()
            .add_algo(
                "do_nothing",
                |g: AdjArrayIn, buffer, iteration, total_iterations| {
                    buffer.write(
                        format!("iteration_{}_of_{}", iteration, total_iterations),
                        "",
                    );
                    (0..g.number_of_nodes()).collect_vec()
                },
            )
            .strict(false)
            .iterations(Iterations::new(
                3,
                InnerIterations::Adaptive(std::time::Duration::new(0, 500_000_000), 1),
            ))
            .split_into_sccs(false)
            .run(output_path.clone())
            .unwrap();

        let output = read_bench_output(output_path);
        assert_eq!(output.len(), 4);
        for i in 1..4 {
            // Algo does nothing except write to a buffer, min iterations is 1, min seconds is 500 ms, so number of
            // iterations should be at least 10
            assert!(output[i].len() >= 10);
        }
    }

    #[test]
    fn test_fvs_file_writer() {
        let fvs_dir_path = tempfile::TempDir::new().unwrap().into_path();

        FvsBench::new()
            .add_graph("4_n", AdjArrayIn::from(&[(0, 0), (1, 1), (2, 2), (3, 3)]))
            .add_graph("one_node", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |_, _, _, _| vec![])
            .add_algo("test_algo", |g, _, _, _| {
                (0..g.number_of_nodes()).collect_vec()
            })
            .reduce_graphs(false)
            .fvs_output_dir(fvs_dir_path.clone())
            .num_threads(1)
            .run_with_writer(sink())
            .unwrap();

        let fvs_paths = fvs_dir_path
            .read_dir()
            .unwrap()
            .map(|r| r.unwrap().path())
            .sorted()
            .collect_vec();

        let fvs_file_names = fvs_paths
            .iter()
            .map(|path_buf| path_buf.file_name().unwrap())
            .collect_vec();

        assert_eq!(
            fvs_file_names,
            vec![
                "4_n_k_0.fvs",
                "4_n_k_4.fvs",
                "one_node_k_0.fvs",
                "one_node_k_1.fvs",
            ]
        );

        let fvs_results = fvs_paths
            .iter()
            .map(|path_buf| {
                let mut fvs = String::new();
                File::open(path_buf)
                    .unwrap()
                    .read_to_string(&mut fvs)
                    .unwrap();
                fvs
            })
            .collect_vec();

        assert_eq!(fvs_results, vec!["[]", "[0, 1, 2, 3]", "[]", "[0]"]);
    }
}
