use dfvs::graph::io::MetisRead;
use dfvs::graph::*;
use dfvs::pre_processor_reduction::*;
use glob::glob;
use log::LevelFilter;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    dfvs::log::build_pace_logger_for_level(LevelFilter::Info);

    let mut total_n = 0;
    let mut total_m = 0;

    for filename in glob("data/pace/h*/*.metis").unwrap().filter_map(Result::ok) {
        // start timer
        let start = Instant::now();

        // read in file
        let file_in = File::open(filename.as_path())?;
        let buf_reader = BufReader::new(file_in);
        let graph = AdjArrayUndir::try_read_metis(buf_reader)?;
        let n = graph.number_of_nodes();
        let m = graph.number_of_edges();

        // apply reduction rules
        let mut state = PreprocessorReduction::from(graph);
        state.apply_rules_exhaustively(false);
        // remove all disconnected vertices
        let out_graph = state.graph().remove_disconnected_verts().0;
        // report performance
        println!(
            "{:<38} | in n={:>7} m={:>8} | out n={:>7} m={:>8} | fvs: {:>6} | scc: {:>4} | time: {:>5}ms",
            filename.display(),
            n,
            m,
            out_graph.number_of_nodes(),
            out_graph.number_of_edges(),
            state.fvs().len(),
            out_graph.strongly_connected_components_no_singletons().len(),
            start.elapsed().as_millis()
        );

        total_n += out_graph.number_of_nodes();
        total_m += out_graph.number_of_edges();
    }

    println!("{}, {}", total_n, total_m);

    Ok(())
}
