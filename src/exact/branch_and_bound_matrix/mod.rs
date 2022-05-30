use crate::algorithm::*;
use crate::graph::*;
use crate::utils::*;
use itertools::Itertools;
use num::cast::AsPrimitive;
use num::PrimInt;

#[cfg(target_arch = "x86_64")]
mod avx2;
pub mod bb_core;
pub mod bb_graph;
pub mod bb_stats;
mod generic_int_graph;
mod scc_iterator;

pub use bb_core::*;
pub use bb_graph::*;
pub use generic_int_graph::*;
use scc_iterator::*;

use bb_stats::BBStats;

pub struct BranchAndBoundMatrix<'a, G> {
    graph: &'a G,
    solution: Option<Vec<Node>>,
}

impl<'a, G> BranchAndBoundMatrix<'a, G>
where
    G: 'a + AdjacencyList,
{
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            solution: None,
        }
    }
}

impl<'a, G> IterativeAlgorithm for BranchAndBoundMatrix<'a, G>
where
    G: 'a + AdjacencyList,
{
    fn execute_step(&mut self) {
        self.solution = branch_and_bound_matrix(self.graph, None);
        assert!(self.solution.is_some());
    }

    fn is_completed(&self) -> bool {
        self.solution.is_some()
    }

    fn best_known_solution(&mut self) -> Option<&[Node]> {
        self.solution.as_deref()
    }
}

impl<'a, G> TerminatingIterativeAlgorithm for BranchAndBoundMatrix<'a, G> where G: 'a + AdjacencyList
{}

/// Return the smallest dfvs with up to `upper_bound` nodes (inclusive).
pub fn branch_and_bound_matrix<G: AdjacencyList>(
    graph: &G,
    upper_bound: Option<Node>,
) -> Option<Vec<Node>> {
    branch_and_bound_matrix_stats(graph, upper_bound, &mut BBStats::new())
}

pub fn branch_and_bound_matrix_stats<G: AdjacencyList>(
    graph: &G,
    upper_bound: Option<Node>,
    stats: &mut BBStats,
) -> Option<Vec<Node>> {
    fn solution_to_vec<T: IntegerIterators>(s: Option<T>) -> Option<Vec<Node>> {
        s.map(|s| s.iter_ones().map(|x| x as Node).collect_vec())
    }

    let upper_bound = upper_bound.unwrap_or_else(|| graph.number_of_nodes()) + 1;
    if graph.len() > 64 {
        let graph = Graph128::from(graph);
        solution_to_vec(branch_and_bound_impl_sccs(&graph, 0, upper_bound, stats))
    } else if graph.len() > 32 {
        let graph = Graph64::from(graph);
        solution_to_vec(branch_and_bound_impl_sccs(&graph, 0, upper_bound, stats))
    } else if graph.len() > 16 {
        let graph = Graph32::from(graph);
        solution_to_vec(branch_and_bound_impl_sccs(&graph, 0, upper_bound, stats))
    } else if graph.len() > 8 {
        let graph = Graph16::from(graph);
        solution_to_vec(branch_and_bound_impl_sccs(&graph, 0, upper_bound, stats))
    } else {
        let graph = Graph8::from(graph);
        solution_to_vec(branch_and_bound_impl_sccs(&graph, 0, upper_bound, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::BitSet;
    use crate::random_models::gnp::generate_gnp;
    use crate::random_models::planted_cycles::generate_planted_cycles;
    use rand::prelude::SliceRandom;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn bb() {
        {
            // graph is acyclic -> solution {}
            assert_eq!(
                branch_and_bound_matrix(&AdjListMatrix::from(&[(0, 1)]), None).unwrap(),
                vec![]
            );
        }

        {
            // graph has loop at 0 -> solution {0}
            assert_eq!(
                branch_and_bound_matrix(&AdjListMatrix::from(&[(0, 1), (0, 0)]), None,).unwrap(),
                vec![0]
            );
        }

        {
            // graph has loop at 0 -> solution {0}
            assert_eq!(
                branch_and_bound_matrix(&AdjListMatrix::from(&[(0, 1), (0, 0)]), None,).unwrap(),
                vec![0]
            );
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            assert_eq!(
                branch_and_bound_matrix(&AdjListMatrix::from(&[(0, 1), (0, 0), (3, 3)]), None,)
                    .unwrap(),
                vec![0, 3]
            );
        }

        {
            // no solution, as limit too low
            assert!(branch_and_bound_matrix(
                &AdjListMatrix::from(&[(0, 1), (0, 0), (3, 3)]),
                Some(1),
            )
            .is_none());
        }

        {
            // graph has loop at 2 -> solution {2}
            let graph = AdjListMatrix::from(&[(0, 1), (1, 2), (2, 3), (3, 0), (2, 2)]);
            assert_eq!(branch_and_bound_matrix(&graph, None).unwrap(), vec![2]);
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            let graph = AdjListMatrix::from(&[(0, 0), (1, 2), (2, 3), (3, 4), (4, 1), (3, 3)]);
            assert_eq!(branch_and_bound_matrix(&graph, None).unwrap(), vec![0, 3]);
        }

        {
            // graph has loop at 0, 3 -> solution {0, 3}
            let mut nodes = [0, 1, 2, 3, 4, 5];
            let graph = AdjListMatrix::from(&[(0, 0), (1, 2), (2, 3), (3, 4), (4, 1), (5, 5)]);

            for _ in 0..10 {
                nodes.shuffle(&mut rand::thread_rng());
                let solution = branch_and_bound_matrix(&graph, None).unwrap();
                assert_eq!(solution.len(), 3);
                assert_eq!(solution[0], 0);
                assert!(1 <= solution[1] && solution[1] < 5);
                assert_eq!(solution[2], 5);
            }
        }
    }

    #[test]
    fn bb_scc_specific() {
        // limit reached in first scc
        assert!(branch_and_bound_matrix(
            &AdjListMatrix::from(&[(0, 1), (1, 0), (2, 3), (3, 2)]),
            Some(1),
        )
        .is_none());

        // limit reached in second scc
        assert!(branch_and_bound_matrix(
            &AdjListMatrix::from(&[
                (0, 1),
                (1, 0),
                (2, 3),
                (2, 4),
                (3, 3),
                (3, 4),
                (4, 2),
                (5, 5)
            ]),
            Some(3),
        )
        .is_none());

        assert_eq!(
            branch_and_bound_matrix(
                &AdjListMatrix::from(&[(0, 1), (0, 2), (1, 1), (1, 2), (2, 0),]),
                Some(3),
            )
            .unwrap()
            .len(),
            2
        );

        assert_eq!(
            branch_and_bound_matrix(
                &AdjListMatrix::from(&[
                    (0, 1),
                    (1, 0),
                    (2, 3),
                    (2, 4),
                    (3, 3),
                    (3, 4),
                    (4, 2),
                    (5, 5)
                ]),
                Some(4),
            )
            .unwrap()
            .len(),
            4
        );

        assert_eq!(
            branch_and_bound_matrix(
                &AdjListMatrix::from(&[
                    (0, 3),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (2, 4),
                    (3, 1),
                    (4, 0),
                    (4, 2),
                ]),
                None,
            )
            .unwrap()
            .len(),
            2
        );

        assert_eq!(
            branch_and_bound_matrix(
                &AdjListMatrix::from(&[(0, 3), (1, 0), (1, 2), (1, 3), (2, 4), (3, 1), (4, 2),]),
                None,
            )
            .unwrap()
            .len(),
            2
        );

        // several recursive sccs when removing a node
        assert_eq!(
            branch_and_bound_matrix(
                &AdjListMatrix::from(&[
                    (0, 1),
                    (1, 4),
                    (2, 1),
                    (2, 3),
                    (2, 4),
                    (3, 5),
                    (4, 2),
                    (5, 3),
                    (5, 0)
                ]),
                None,
            )
            .unwrap()
            .len(),
            2
        );
    }

    #[test]
    fn bb_generated_tests() {
        // The results were generated by the branch_and_bound implementation in MR19.
        let solution_sizes = vec![
            4, 4, 5, 5, 4, 4, 6, 7, 5, 6, 5, 6, 6, 8, 6, 6, 8, 6, 5, 6, 7, 7, 7, 5, 7, 8, 9, 9, 7,
            7, 9, 9, 8, 10, 8, 8,
        ];

        let mut gen = Pcg64Mcg::seed_from_u64(123);

        for n in 10..=21 {
            for avg_deg in [5] {
                for i in 0..3 {
                    let p = avg_deg as f64 / n as f64;
                    let mut graph: AdjArray = generate_gnp(&mut gen, n, p);

                    for i in graph.vertices_range() {
                        graph.try_remove_edge(i, i);
                    }

                    let solution = branch_and_bound_matrix(&graph, None).unwrap();
                    let solution_mask = {
                        let mut set = BitSet::new_all_set(graph.len());
                        for node in &solution {
                            set.unset_bit(*node as usize);
                        }
                        set
                    };

                    assert_eq!(solution.len(), solution_sizes[((n - 10) * 3 + i) as usize]);
                    assert!(graph.vertex_induced(&solution_mask).0.is_acyclic());
                }
            }
        }
    }

    #[test]
    fn planted_cycles() {
        let mut gen = Pcg64Mcg::seed_from_u64(234);

        for k in [1, 3] {
            for num_cycles in [k, 2 * k] {
                for cycle_len in [1, 2] {
                    for offset in [1, 5] {
                        let n = num_cycles * cycle_len + k + offset;
                        for avg_deg in [1.5, 5.0] {
                            if 2.0 * avg_deg >= n as f64 {
                                continue;
                            }

                            let (graph, solution): (AdjArray, Vec<_>) = generate_planted_cycles(
                                &mut gen, n, avg_deg, k, num_cycles, cycle_len,
                            );
                            let computed = branch_and_bound_matrix(&graph, None).unwrap();

                            assert_eq!(solution.len(), computed.len());
                        }
                    }
                }
            }
        }
    }
}
