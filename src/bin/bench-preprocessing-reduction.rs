use dfvs::bitset::BitSet;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::PaceRead;
use dfvs::graph::*;
use dfvs::pre_processor_reduction::*;
use glob::glob;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    dfvs::log::build_pace_logger();

    for filename in glob("data/netrep/*/*").unwrap().filter_map(Result::ok) {
        // start timer
        let start = Instant::now();

        // read in file
        let file_in = File::open(filename.as_path())?;
        let buf_reader = BufReader::new(file_in);
        let graph = AdjArrayIn::try_read_pace(buf_reader)?;
        let n = graph.number_of_nodes();
        let m = graph.number_of_edges();

        // apply reduction rules
        let mut state = PreprocessorReduction::from(graph);
        state.apply_rules_exhaustively();

        // remove all disconnected vertices
        let out_graph = {
            let graph = state.graph();
            let bs = BitSet::new_all_unset_but(
                graph.len(),
                graph.vertices().filter(|&u| graph.total_degree(u) > 0),
            );
            graph.vertex_induced(&bs).0
        };

        // report performance
        println!(
            "{:<40} | in n={:>7} m={:>8} | out n={:>7} m={:>8} | time: {:>5}ms",
            filename.display(),
            n,
            m,
            out_graph.number_of_nodes(),
            out_graph.number_of_edges(),
            start.elapsed().as_millis()
        );
    }

    Ok(())
}
