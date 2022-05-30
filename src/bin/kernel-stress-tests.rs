use dfvs::exact::branch_and_bound_matrix::branch_and_bound_matrix;
use dfvs::graph::io::*;
use dfvs::graph::*;
use dfvs::kernelization::{Rules, SuperReducer};
use glob::glob;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let filenames = glob("data/stress_kernels/*.metis")
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let start = Instant::now();
    let graphs: Vec<_> = filenames
        .into_par_iter()
        .filter_map(|filename| {
            if filename.to_str()?.replace("_kernel", "").contains("_k") {
                return None;
            }

            let file_extension = filename.extension().unwrap().to_str().unwrap().to_owned();
            let file_format = FileFormat::from_str(&file_extension).ok()?;
            Some((
                String::from(filename.to_str().unwrap()),
                AdjArrayUndir::try_read_graph(file_format, filename).ok()?,
            ))
        })
        .filter(|(_, graph)| graph.len() < 120)
        .map(|(filename, graph)| {
            let k = branch_and_bound_matrix(&graph, None).unwrap().len();
            (filename, graph, k)
        })
        .collect();

    println!(
        "Loaded {} graphs in {} ms",
        graphs.len(),
        start.elapsed().as_millis()
    );

    let progress = Arc::new(AtomicUsize::new(0));

    graphs
        .par_iter()
        .for_each(|(filename, graph, k)| {
            use dfvs::kernelization::Rules::*;
            let rules = vec![
                Rule5(1000),
                DiClique,
                CompleteNode,
                PIE,
                DOME,
                DOMN,
                C4,
                //Crown(1000),
                Unconfined,
                RedundantCycles,
                Dominance,
            ];

            let mut accepted = HashSet::with_capacity(1000);
            accepted.insert((graph.digest_sha256(),0) );

            let mut process_graph = |r : &Vec<Rules>| {
                let mut reducer = SuperReducer::with_settings(graph.clone(), r.clone(), false);
                let (kernel_solution, sccs) = reducer.reduce().unwrap();
                assert_eq!(sccs.len(), 1);

                let scc = sccs[0].0.clone();
                let digest = scc.digest_sha256();

                if accepted.contains(&(digest.clone(), kernel_solution.len()) ) {
                    return;
                }

                let solution = branch_and_bound_matrix(&scc, None).unwrap();

                let reduced_k = kernel_solution.len() + solution.len();

                if reduced_k != *k {
                    println!(
                        "rule = {:?}, filename = {:>100}, k = {:>5}, k-red = {:>5} | kernel = {:?}, solution = {:?}",
                        r,
                        filename,
                        *k,
                        reduced_k,
                        &kernel_solution,
                        &solution
                    );
                }

                accepted.insert((digest, kernel_solution.len()) );

                assert!(reduced_k >= *k);
            };

            rules.into_iter().powerset().for_each(|mut r| {
                let k = r.len();
                if k < 4 {
                    r.into_iter().permutations(k).for_each(|p| process_graph(&p));
                } else {
                    for _ in 0..40 {
                        r.shuffle(&mut thread_rng());
                        process_graph(&r);
                    }
                }
            });

            let progress = progress.clone();
            let num_completed= progress.fetch_add(1, Ordering::SeqCst);
            if  num_completed >0 && num_completed % 20 == 0 {
                println!("Completed {} graphs", num_completed);
            }
        });

    Ok(())
}
