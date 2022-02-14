use dfvs::exact::branch_and_bound::branch_and_bound;
use dfvs::graph::{AdjArray, GraphEdgeEditing, GraphOrder};
use dfvs::random_models::gnp::generate_gnp;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    #[cfg(feature = "pace-logging")]
    dfvs::log::build_pace_logger();

    let mut gen = Pcg64Mcg::seed_from_u64(123);

    for n in 10..31 {
        for avg_deg in [5] {
            let start = Instant::now();
            let mut num = 0;
            let mut edges = 0;
            let mut size = 0;

            loop {
                let p = avg_deg as f64 / n as f64;
                let mut graph: AdjArray = generate_gnp(&mut gen, n, p);

                for i in graph.vertices_range() {
                    graph.try_remove_edge(i, i);
                }

                edges += graph.number_of_edges();

                let solution = branch_and_bound(&graph, None).unwrap();
                size += solution.len();

                num += 1;
                if num > 5 && start.elapsed().as_secs() > 1 {
                    break;
                }
            }

            let iters = num as f64;
            println!(
                "n = {}, m = {}, size = {} iters = {}, time = {} ms",
                n,
                edges as f64 / iters,
                size as f64 / iters,
                iters,
                start.elapsed().as_millis() as f64 / iters
            );
        }
    }

    Ok(())
}
