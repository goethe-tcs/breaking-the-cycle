use log::{debug, info, trace};
use std::fmt::{Debug, Display};
use std::io::{ErrorKind, Result};
use std::marker::PhantomData;
use std::path::Path;
use std::time::Instant;

use super::io::keyed_csv_writer::KeyedCsvWriter;
use super::io::keyed_writer::KeyedWriter;
use crate::bench::io::bench_writer::{BenchWriter, DualWriter};
use crate::graph::io::{FileFormat, GraphRead};
use crate::graph::{
    AdjacencyListIn, AdjacencyTest, Getter, GraphEdgeEditing, GraphNew, GraphOrder,
    InducedSubgraph, Node, Traversal,
};
use crate::pre_processor_reduction::{PreprocessorReduction, ReductionState};

/// Measures the computation time and solution size of multiple DFVS-algorithms against multiple
/// graphs. Each algorithm is run for each graph at least once.
pub struct FvsBench<G, W> {
    strict: bool,
    reduce_graphs: bool,
    iterations: usize,

    /// Provide (load/create/etc) graphs that are passed into the DFVS-algorithms. Each graph will
    /// be processed by every algorithm at least once
    graph_providers: Vec<LabeledGraphProvider<G>>,

    /// DFVS algorithms with a label
    algos: Vec<LabeledFvsAlgo<G, W>>,

    /// Enables using W as a generic type parameter, which helps the compiler to determine the type
    /// of the BenchWriter of algorithm closures, when passing algorithms into the benchmark.
    _phantom_writer: PhantomData<W>,
}
type GraphProvider<G> = dyn Fn() -> Result<G>;
type LabeledGraphProvider<G> = (String, Box<GraphProvider<G>>);

type FvsAlgo<G, W> = dyn Fn(G, &mut W) -> Vec<Node>;
type LabeledFvsAlgo<G, W> = (String, Box<FvsAlgo<G, W>>);

impl<G, W> FvsBench<G, W>
where
    G: GraphOrder
        + AdjacencyListIn
        + AdjacencyTest
        + GraphEdgeEditing
        + GraphNew
        + GraphRead
        + Debug
        + Clone
        + 'static,
    W: BenchWriter,
{
    pub fn new() -> Self {
        Self {
            strict: false,
            reduce_graphs: true,
            iterations: 1,

            graph_providers: vec![],
            algos: vec![],
            _phantom_writer: Default::default(),
        }
    }

    /// **Only use this method for small test graphs!**
    ///
    /// Use `Bench::add_graph_file` or `Bench::add_graph_provider()` instead, to delay loading
    /// the graph into memory until it is needed by the benchmark and can be freed by the
    /// benchmark when it is no longer needed.
    pub fn add_graph(&mut self, label: impl Display, graph: G) -> &mut Self {
        self.add_graph_provider(label, move || Ok(graph.clone()))
    }

    /// Adds a graph file that will be loaded by the benchmark.
    pub fn add_graph_file(
        &mut self,
        file_format: FileFormat,
        file_path: impl AsRef<Path>,
    ) -> Result<&mut Self> {
        let create_label_err =
            || std::io::Error::new(ErrorKind::Other, "Failed to retrieve name of graph file!");

        let file_path = file_path.as_ref().to_path_buf();
        let file_path_closure = file_path.clone();
        let graph_provider = move || G::try_read_graph(file_format, &file_path_closure);

        let file_name = file_path
            .file_name()
            .ok_or_else(create_label_err)?
            .to_str()
            .ok_or_else(create_label_err)?;

        Ok(self.add_graph_provider(file_name, graph_provider))
    }

    /// Adds a function that provides (loads/creates/etc) a graph. The function will be called by
    /// the benchmark and the graph is passed into the DFVS-algorithms.
    pub fn add_graph_provider<F>(&mut self, label: impl Display, provider: F) -> &mut Self
    where
        F: Fn() -> Result<G> + 'static,
    {
        self.graph_providers
            .push((label.to_string(), Box::new(provider)));
        self
    }

    pub fn add_algo<F>(&mut self, label: impl Display, algo: F) -> &mut Self
    where
        F: Fn(G, &mut W) -> Vec<Node> + 'static,
    {
        self.algos.push((label.to_string(), Box::new(algo)));
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
    pub fn iterations(&mut self, value: usize) -> &mut Self {
        assert_ne!(value, 0, "Iterations must be greater than zero!");
        self.iterations = value;
        self
    }

    /// Runs the benchmark using the passed in BenchWriter
    pub fn run_with_writer(&self, writer: &mut W) -> Result<&Self> {
        use std::io::Error;

        let error_factory = |msg| Err(Error::new(ErrorKind::InvalidInput, msg));
        if self.algos.is_empty() {
            return error_factory("No algorithms provided for benchmarking!");
        }
        if self.graph_providers.is_empty() {
            return error_factory("No graphs provided for benchmarking!");
        }

        let mut design_point_i = 0;
        let design_points_total = self.algos.len() * self.graph_providers.len() * self.iterations;
        info!(
            "Running bench with {} design points...",
            design_points_total
        );
        for (graph_i, (graph_label, graph_provider)) in self.graph_providers.iter().enumerate() {
            debug!(
                "graph '{}' ({}/{})...",
                graph_label,
                graph_i,
                self.graph_providers.len()
            );
            debug!("Running graph provider...");
            let input_graph = graph_provider()?;
            trace!("input graph: {:?}", input_graph);

            // TODO: Pass reduction rules in as 'graph-pre-processing-step' instead of using
            //  apply_rules param

            // reduce graph
            let mut state = PreprocessorReduction::from(input_graph.clone());
            if self.reduce_graphs {
                state.apply_rules_exhaustively();
                trace!("reduced graph: {:?}", state.graph());
            }
            let (reduced_graph, reduction_mapper) = state.graph().remove_disconnected_verts();

            // run all algorithms with the graph
            for (algo_label, algo) in self.algos.iter() {
                // benchmark each algorithm/graph pair 'iterations' amount of times
                for _ in 0..self.iterations {
                    design_point_i += 1;
                    let mut total_fvs = state.fvs().to_vec();
                    let mut input_graph = input_graph.clone();
                    let reduced_graph = reduced_graph.clone();
                    writer.write("id", design_point_i)?;
                    debug!("design point {}/{}...", design_point_i, design_points_total);

                    // write graph related metrics
                    writer.write("graph", graph_label)?;
                    writer.write("n", input_graph.len())?;
                    writer.write("m", input_graph.number_of_edges())?;
                    writer.write("n_reduced", reduced_graph.len())?;
                    writer.write("m_reduced", reduced_graph.number_of_edges())?;

                    // run single algorithm
                    let start = Instant::now();
                    let algo_fvs: Vec<Node> = algo(reduced_graph, writer);
                    let elapsed_time = start.elapsed().as_secs_f64();

                    // add algo fvs to the reduction fvs
                    let remapped_algo_fvs = algo_fvs
                        .iter()
                        .map(|id| reduction_mapper.old_id_of(*id).unwrap_or(*id));
                    total_fvs.extend(remapped_algo_fvs);

                    // write algorithm related metrics
                    writer.write("algo", algo_label)?;
                    writer.write("fvs_reduction", state.fvs().len())?;
                    writer.write("fvs_algo", algo_fvs.len())?;
                    writer.write("fvs_total", total_fvs.len())?;
                    writer.write("elapsed_sec_algo", elapsed_time)?;

                    // apply fvs to graph (to print resulting graph & check if graph is acyclic)
                    input_graph.remove_edges_of_nodes(total_fvs.iter());
                    trace!("fvs_reduction: {:?}", state.fvs());
                    trace!("fvs_algo: {:?}", algo_fvs);
                    trace!("fvs_total: {:?}", total_fvs);
                    trace!("resulting_graph: {:?}", input_graph);

                    // check if result is acyclic
                    let is_acyclic = input_graph.is_acyclic();
                    writer.write("is_result_acyclic", is_acyclic)?;
                    if self.strict && !is_acyclic {
                        writer.end_design_point()?;
                        writer.end_graph_section()?;
                        return Err(Error::new(
                            ErrorKind::Other,
                            format!(
                                "Result of design point {} (algorithm '{}') was not acyclic!",
                                design_point_i, algo_label
                            ),
                        ));
                    }

                    writer.end_design_point()?;
                }
            }
            writer.end_graph_section()?;
        }

        info!("Bench finished");
        Ok(self)
    }
}

impl<G, W> Default for FvsBench<G, W>
where
    G: GraphOrder
        + AdjacencyListIn
        + AdjacencyTest
        + GraphEdgeEditing
        + GraphNew
        + GraphRead
        + Debug
        + Clone
        + 'static,
    W: BenchWriter,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G> FvsBench<G, DualWriter>
where
    G: GraphOrder
        + AdjacencyListIn
        + AdjacencyTest
        + GraphEdgeEditing
        + GraphNew
        + GraphRead
        + Debug
        + Clone
        + 'static,
{
    /// Runs the benchmark using a DualWriter that writes bench results to standard output as well
    /// as to a csv file
    pub fn run(&self, csv_path: impl AsRef<Path>) -> Result<&Self> {
        let mut writer = DualWriter::new(
            KeyedWriter::new_std_out_writer(),
            KeyedCsvWriter::new(csv_path.as_ref().to_path_buf())?,
        );
        self.run_with_writer(&mut writer)
    }
}

#[cfg(test)]
mod tests_bench {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;

    #[test]
    #[should_panic]
    fn test_zero_iterations_panic() {
        FvsBench::new()
            .iterations(0)
            .add_graph("self_loop", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |_, _| vec![])
            .run_with_writer(&mut ())
            .unwrap();
    }

    #[test]
    fn test_strict_no_error() {
        FvsBench::default()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .add_algo("test_algo", |_, _| vec![0])
            .strict(true)
            .reduce_graphs(false)
            .run_with_writer(&mut ())
            .unwrap();
    }

    #[test]
    fn test_strict_error() {
        let error_kind = FvsBench::default()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .add_algo("test_algo", |_, _| vec![])
            .strict(true)
            .reduce_graphs(false)
            .run_with_writer(&mut ())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::Other);
    }

    #[test]
    fn test_zero_graphs_error() {
        let error_kind = FvsBench::new()
            .add_algo("test_algo", |_graph: AdjArrayIn, _| vec![])
            .run_with_writer(&mut ())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn test_zero_algos_error() {
        let error_kind = FvsBench::new()
            .add_graph("test_graph", AdjArrayIn::from(&vec![(0, 0)]))
            .run_with_writer(&mut ())
            .err()
            .unwrap()
            .kind();

        assert_eq!(error_kind, ErrorKind::InvalidInput);
    }
}

#[cfg(feature = "tempfile")]
#[cfg(test)]
mod tests_bench_with_tempfile {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use csv::Reader;
    use itertools::Itertools;

    use crate::graph::io::PaceWrite;
    use std::fs::File;

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
                vec[10] = "".to_string(); // remove unstable elapsed time entry
                vec
            });

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
            "algo",
            "fvs_reduction",
            "fvs_algo",
            "fvs_total",
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
            .add_algo("test_algo", |_, _| vec![1, 2])
            .run(output_path.clone())
            .unwrap();

        let actual_output = read_bench_output(output_path);
        let expected_output = vec![
            get_bench_column_headers(),
            vec![
                "1",
                "test_graph",
                "4",
                "7",
                "3",
                "6",
                "test_algo",
                "1",
                "2",
                "3",
                "",
                "true",
            ],
        ];
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_graph_provider() {
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
            .add_graph_provider("self_loops", || {
                Ok(AdjArrayIn::from(&vec![(0, 0), (1, 1), (2, 2), (3, 3)]))
            })
            .add_graph("tiny_graph", AdjArrayIn::from(&vec![(0, 0), (1, 2)]))
            .add_algo("do_nothing", |_graph: AdjArrayIn, _| vec![])
            .strict(false)
            .reduce_graphs(false)
            .run_with_writer(&mut KeyedCsvWriter::new(output_path.clone()).unwrap())
            .unwrap();

        let actual_output = read_bench_output(output_path);

        let expected_output = vec![
            get_bench_column_headers(),
            vec![
                "1",
                "test_graph_provider_graph",
                "2",
                "2",
                "2",
                "2",
                "do_nothing",
                "0",
                "0",
                "0",
                "",
                "false",
            ],
            vec![
                "2",
                "self_loops",
                "4",
                "4",
                "4",
                "4",
                "do_nothing",
                "0",
                "0",
                "0",
                "",
                "false",
            ],
            vec![
                "3",
                "tiny_graph",
                "3",
                "2",
                "3",
                "2",
                "do_nothing",
                "0",
                "0",
                "0",
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
            .add_graph_provider("5_n", || Ok(AdjArrayIn::new(5)))
            .add_graph("self_loop", AdjArrayIn::from(&[(0, 0)]))
            .add_algo("do_nothing", |_, _| vec![])
            .add_algo("test_algo", |_, _| vec![0])
            .strict(false)
            .iterations(3)
            .run_with_writer(&mut KeyedCsvWriter::new(output_path.clone()).unwrap())
            .unwrap();

        let output = read_bench_output(output_path);
        assert_eq!(output.len(), 19);
    }
}
