use dfvs::bench::fvs_bench::FvsBench;
use dfvs::bench::io::bench_dir;
use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::heuristics::greedy::{greedy_dfvs, MaxDegreeSelector};
use dfvs::log::build_pace_logger;
use dfvs::random_models::gnp::generate_gnp;
use log::info;
use rand::prelude::*;
use rand_pcg::Pcg64;

fn main() -> std::io::Result<()> {
    build_pace_logger();

    let seed = 1;
    let mut rng = Pcg64::seed_from_u64(seed);
    info!("Using seed {}", seed);

    info!("Creating graphs...");
    let mut bench = FvsBench::new();
    for n in [6000, 12000, 24000] {
        let label = format!("rnd_n_{}k", n / 1000);
        let p = (n as f64).log(2.0) / (4.0 * (n as f64));
        bench.add_graph(label, generate_gnp::<AdjArrayIn, _>(&mut rng, n, p));
    }

    bench.add_algo("greedy", |graph, _| {
        greedy_dfvs::<MaxDegreeSelector<_>, _, _>(graph)
    });

    bench
        .strict(false)
        .iterations(3)
        .run(bench_dir()?.join("heuristics.csv"))?;

    Ok(())
}
