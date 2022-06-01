#![allow(rustdoc::private_intra_doc_links)]

use crate::algorithm::*;
use crate::graph::*;
use itertools::Itertools;
use log::info;
use std::fmt::Debug;

mod frame;
pub mod result_cache;

use crate::bitset::BitSet;
use crate::exact::branch_and_bound_matrix::branch_and_bound_matrix;
use frame::*;
use result_cache::ResultCache;

/// This implementation is a classical branch-and-bound algorithm. In each step, we apply
/// kernelization rules, may compute lower-bounds, and then try several branching strategies.
/// If an output is produced, it is guaranteed to be a minimal directed feedback vertex set.
///
/// We implement this recursive algorithm using an explicit call stack. This design avoids
/// stack overflows and simplifies checks and profiling. The bulk of the work is carried out in
/// [`Frame`] which emulates a stack frame. A frame has two entry points:
/// [`Frame::initialize`] is called upon the first call. It may either return
/// [`BBResult::Branch`]`(child`) in which case, we will put `child` on the stack and call it's
/// `initialize`. Alternatively, a call may return [``BBResult::Result``] which is then presented
/// to the frame's parent by calling the parent's [`Frame::resume`] method.
/// If no parent exists, the result is returned to the calling code. A stack frame can be resumed
/// an arbitrary number of times.
///
/// We implement the [`IterativeAlgorithm`] trait in order to support preemption. A step in the
/// iterative algorithm corresponds to call of the current frame's `initialize` or `resume`.
///
/// # Info
/// The recommended approach to use this algorithm is by calling
/// `BranchAndBound::new(graph).run_to_completion()`. You may also replace the `new` constructor
/// with the `with_paranoid` constructor, which behaves analogously but carries out additional
/// after each stack frame completed. This comes with increased memory (roughly doubled) and
/// runtime costs.
///
/// # Example
/// ```
/// use dfvs::algorithm::*;
/// use dfvs::graph::*;
/// use dfvs::exact::BranchAndBound;
///
/// let input_graph = AdjArrayUndir::from(&[(0,1), (1,0), (1,2), (2, 1)]);
/// let solution = BranchAndBound::new(input_graph).run_to_completion().unwrap();
/// assert_eq!(solution, vec![1]);
/// ```
pub struct BranchAndBound<G: BnBGraph> {
    stack: Vec<Frame<G>>,
    paranoia_stack: Vec<G>,
    from_child: Option<OptSolution>,
    solution: Option<OptSolution>,

    cache: ResultCache,

    number_of_nodes: Node,
    iterations: usize,

    paranoid: bool,
    super_verbose: bool,
    drop_output: bool,
    cross_validation: bool,
}

pub trait BnBGraph:
    AdjacencyListIn
    + AdjacencyListUndir
    + AdjacencyTest
    + AdjacencyTestUndir
    + GraphNew
    + GraphEdgeEditing
    + Clone
    + Default
    + Debug
{
}

impl<
        G: AdjacencyListIn
            + AdjacencyListUndir
            + AdjacencyTest
            + AdjacencyTestUndir
            + GraphNew
            + GraphEdgeEditing
            + Clone
            + Default
            + Debug,
    > BnBGraph for G
{
}

enum BBResult<G> {
    Result(OptSolution),
    Branch(Frame<G>),
}

pub type Solution = Vec<Node>;
pub type OptSolution = Option<Solution>;

const MIN_NODES_FOR_CACHE: Node = 16;

impl<G: BnBGraph> BranchAndBound<G> {
    pub fn new(graph: G) -> Self {
        let mut stack: Vec<Frame<G>> = Vec::with_capacity(3 * (graph.len() + 2));
        let number_of_nodes = graph.number_of_nodes();
        stack.push(Frame::new(graph, 0, number_of_nodes + 1));

        Self {
            stack,
            paranoia_stack: Vec::new(),
            solution: None,
            from_child: None,
            number_of_nodes,
            iterations: 0,
            paranoid: false,
            super_verbose: false,
            drop_output: false,
            cross_validation: false,
            cache: Default::default(),
        }
    }

    pub fn with_paranoia(graph: G) -> Self {
        let mut graph_stack = Vec::with_capacity(3 * (graph.len() + 2));
        graph_stack.push(graph.clone());
        let mut res = Self::new(graph);
        res.paranoid = true;
        res.paranoia_stack = graph_stack;
        res
    }

    /// Sets an inclusive lower bound on the minimum DFVS size. This is a hint which may or may
    /// not be used to prune the search tree. It is undefined behaviour if `lower_bound` exceeds
    /// the size of the minimum DFVS.
    ///
    /// # Warning
    /// This method may only be called before the first execution of the algorithm.
    pub fn set_lower_bound(&mut self, lower_bound: Node) {
        assert_eq!(self.iterations, 0);
        self.stack.last_mut().unwrap().lower_bound = lower_bound;
        self.stack.last_mut().unwrap().initial_lower_bound = lower_bound;
    }

    /// Sets an inclusive upper bound on the minimum DFVS size. This is a hint which may or may
    /// not be used to prune the search tree. If `upper_bound` is smaller than the size of the
    /// minimum DFVS no solution can be produced.
    ///
    /// # Warning
    /// This method may only be called before the first execution of the algorithm.
    pub fn set_upper_bound(&mut self, upper_bound: Node) {
        assert_eq!(self.iterations, 0);
        assert!(upper_bound < self.number_of_nodes);

        self.stack.last_mut().unwrap().upper_bound = upper_bound + 1;
        self.stack.last_mut().unwrap().initial_upper_bound = upper_bound + 1;
    }

    /// Returns the number of recursive calls (i.e. calls direct or indirect calls to
    /// [`BranchAndBound::execute_step`]) processed so far
    pub fn number_of_iterations(&self) -> usize {
        self.iterations
    }

    /// Prints a backtrace if the solver instance is dropped without completing.
    /// This is useful if an assertion fired during the computation. However, be aware,
    /// that if you're using the solver with a time limit, this functionality might also
    /// fire if the solver did not finish within the time budget.
    pub fn set_drop_output(&mut self, enabled: bool) {
        self.drop_output = enabled;
    }

    /// Prints a call stack for each call into frame. Requires paranoia.
    ///
    /// # Warning
    /// This might produces millions lines of output.
    pub fn set_print_call_stack(&mut self, enabled: bool) {
        assert!(self.paranoid || !enabled);
        self.super_verbose = enabled
    }

    /// Uses the matrix solver to cross check all results obtained by the solver for small
    /// graphs. Requires paranoia.
    ///
    /// # Warning
    /// This might be extremely slow! Never use this feature in production.
    pub fn set_cross_validation(&mut self, enabled: bool) {
        assert!(self.paranoid || !enabled);
        self.cross_validation = enabled;
    }

    /// Uses the matrix solver to cross check all results obtained by the solver for small
    /// graphs. Requires paranoia.
    ///
    /// # Warning
    /// This might be extremely slow! Never use this feature in production.
    pub fn set_cache_capacity(&mut self, capacity: usize) {
        self.cache.set_capacity(capacity);
    }
}

impl<G: BnBGraph> TerminatingIterativeAlgorithm for BranchAndBound<G> {}

impl<G: BnBGraph> IterativeAlgorithm for BranchAndBound<G> {
    fn execute_step(&mut self) {
        assert!(self.solution.is_none());

        self.iterations += 1;

        // we execute the last frame on the stack but do not remove it yet, as it will remain there
        // in case it branches
        let mut result = {
            let current = self.stack.last_mut().unwrap();
            if let Some(from_child) = self.from_child.replace(None) {
                current.resume(from_child)
            } else {
                current.initialize()
            }
        };

        // cache look up
        if let BBResult::Branch(frame) = &mut result {
            if self.cache.capacity() > 0 && frame.graph.number_of_nodes() >= MIN_NODES_FOR_CACHE {
                let ub = frame.initial_upper_bound;
                self.from_child = self.cache.get(frame.graph_digest(), ub);
                if let Some(fc) = &mut self.from_child {
                    // we have a cache hit; if it's a result, we have to extend it by the parent's partial result
                    if let Some(res) = fc {
                        res.extend(&frame.partial_solution_parent);
                    }
                    return;
                }
            }
        }

        match result {
            BBResult::Result(res) => {
                assert!(res.is_none() || self.stack.last().unwrap().postprocessor.is_empty());
                let digest = {
                    let frame = self.stack.last_mut().unwrap();
                    if frame.graph.number_of_nodes() >= MIN_NODES_FOR_CACHE {
                        Some(self.stack.last_mut().unwrap().graph_digest().clone())
                    } else {
                        None
                    }
                };

                let frame = self.stack.last().unwrap();

                if self.paranoid {
                    if self.super_verbose {
                        info!(
                            "{:<54} |{} {}",
                            "",
                            (0..self.stack.len()).into_iter().map(|_| ' ').join(""),
                            res.as_ref().map_or(-1, |r| r.len() as i64)
                        );
                    }

                    assert_eq!(self.paranoia_stack.len(), self.stack.len());
                    let graph = self.paranoia_stack.pop().unwrap();

                    if let Some(solution) = res.as_ref() {
                        assert!(solution.iter().all(|&u| u < graph.number_of_nodes()));

                        let solution_mask =
                            BitSet::new_all_set_but(graph.len(), solution.iter().copied());

                        assert!(
                            frame
                                .partial_solution_parent
                                .iter()
                                .all(|&x| !solution_mask[x as usize]) // solution mask is inverted!
                        );

                        // check no repeats
                        assert_eq!(
                            graph.len() - solution_mask.cardinality(), // solution mask is inverted!
                            solution.len(),
                            "Repeat in solution: {:?}",
                            &solution
                        );

                        let induced = graph.vertex_induced(&solution_mask);
                        assert!(induced.0.is_acyclic());
                    }

                    if self.cross_validation && graph.len() < 64 {
                        let res_child = res.as_ref().map(|r| {
                            r.iter()
                                .copied()
                                .filter(|x| !frame.partial_solution_parent.contains(x))
                                .collect_vec()
                        });

                        let rsol = branch_and_bound_matrix(&graph, None).unwrap();
                        assert!(
                            frame.initial_lower_bound <= rsol.len() as Node,
                            "frame initial lower: {} ref: {:?}, sol: {:?}, graph={:?}",
                            frame.initial_lower_bound,
                            rsol,
                            res_child,
                            &graph
                        );
                        if let Some(res_child) = res_child {
                            assert_eq!(
                                rsol.len(),
                                res_child.len(),
                                "ref: {:?}, sol: {:?}, graph={:?}",
                                rsol,
                                res_child,
                                &graph
                            );
                        } else {
                            assert!(
                                rsol.len() as Node >= frame.initial_upper_bound,
                                "ref: {:?}, sol: --, graph={:?}",
                                rsol,
                                &graph
                            );
                        }
                    }
                }

                // make sure that the solution satisfies the required lower/upper bounds;
                // this check is sufficiently cheap that we do it even in the non-paranoid mode
                if let Some(solution) = res.as_ref() {
                    assert!(
                        solution.len()
                            >= frame.initial_lower_bound as usize
                                + frame.partial_solution_parent.len()
                    );
                    assert!(
                        solution.len()
                            < frame.initial_upper_bound as usize
                                + frame.partial_solution_parent.len()
                    );
                }

                if let Some(digest) = digest {
                    self.cache.add_to_cache(
                        digest,
                        res.as_ref().map(|s| {
                            s.iter()
                                .copied()
                                .filter(|x| !frame.partial_solution_parent.contains(x))
                                .collect_vec()
                        }),
                        frame.initial_upper_bound,
                    );
                }

                // frame completed, so it's finally time to remove it from stack
                self.stack.pop();

                if self.stack.is_empty() {
                    self.solution = Some(res);
                    return;
                }

                self.from_child = Some(res);
            }

            BBResult::Branch(frame) => {
                if self.paranoid && self.super_verbose && self.stack.len() < 40 {
                    let graph = &self.paranoia_stack.last().unwrap();
                    let parent = self.stack.last().unwrap();

                    info!(
                        "{:>4} n={:>5} m={:>5} mund={:>5} l={:>5} u={:>5} d={:>4} | {} {}",
                        self.stack.len(),
                        graph.number_of_nodes(),
                        graph.number_of_edges(),
                        graph
                            .vertices()
                            .map(|u| graph.undir_degree(u) as usize)
                            .sum::<usize>()
                            / 2,
                        parent.lower_bound,
                        parent.upper_bound,
                        parent.upper_bound - parent.lower_bound,
                        (0..self.stack.len()).into_iter().map(|_| ' ').join(""),
                        parent.resume_with.describe()
                    );
                }

                assert!(
                    self.stack.len() + 1 < self.stack.capacity(),
                    "stack states: {:?}",
                    self.stack
                        .iter()
                        .map(|x| format!("{:?}\n", &x.resume_with))
                        .join("")
                );
                if self.paranoid {
                    self.paranoia_stack.push(frame.graph.clone());
                }

                self.stack.push(frame);
                self.from_child = None;
            }
        }
    }

    fn is_completed(&self) -> bool {
        self.stack.is_empty()
    }

    fn best_known_solution(&mut self) -> Option<&[Node]> {
        if let Some(sol) = &self.solution {
            sol.as_deref()
        } else {
            None
        }
    }
}

impl<G: BnBGraph> Drop for BranchAndBound<G> {
    fn drop(&mut self) {
        while let Some(frame) = self.stack.pop() {
            if self.drop_output {
                info!(
                    "{} {} (dbg-lower={}, dbg-upper={})",
                    (0..self.stack.len()).into_iter().map(|_| ' ').join(""),
                    frame.resume_with.describe(),
                    frame.initial_lower_bound,
                    frame.initial_upper_bound,
                );
            }
        }

        if self.cache.len() > 100 {
            info!(
                "Cache size: {}, hits: {}, misses: {}",
                self.cache.len(),
                self.cache.number_of_cache_hits(),
                self.cache.number_of_cache_misses()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::BitSet;
    use crate::exact::branch_and_bound_matrix::branch_and_bound_matrix;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::random_models::gnp::generate_gnp;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

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
                    let mut graph: AdjArrayIn = generate_gnp(&mut gen, n, p);

                    for i in graph.vertices_range() {
                        graph.try_remove_edge(i, i);
                    }

                    let solution = BranchAndBound::new(graph.clone())
                        .run_to_completion()
                        .unwrap();
                    let solution_mask =
                        BitSet::new_all_set_but(graph.len(), solution.iter().copied());

                    assert_eq!(solution.len(), solution_sizes[((n - 10) * 3 + i) as usize]);
                    assert!(graph.vertex_induced(&solution_mask).0.is_acyclic());
                }
            }
        }
    }

    #[test]
    fn cross_validation() {
        let mut gen = Pcg64Mcg::seed_from_u64(234);

        for n in 2..7 {
            for avg_deg in [2, 4] {
                if avg_deg + 1 >= n {
                    continue;
                }

                for _ in 0..3 {
                    let p = avg_deg as f64 / n as f64;
                    let graph: AdjArrayIn = generate_gnp(&mut gen, n, p);

                    let sol1 = branch_and_bound_matrix(&graph, None).unwrap();
                    let sol2 = BranchAndBound::new(graph.clone())
                        .run_to_completion()
                        .unwrap();

                    assert_eq!(sol1.len(), sol2.len());

                    let solution_mask = BitSet::new_all_set_but(graph.len(), sol2.iter().copied());
                    assert!(graph.vertex_induced(&solution_mask).0.is_acyclic());
                }
            }
        }
    }
}
