//! All our DFVS algorithms should be implemented using the [`IterativeAlgorithm`] trait.
//!
//! The idea is to allow co-operative threading, i.e. an algorithm does some work (say a few
//! milliseconds) and then breaks to yield  to another algorithm. An external entity (scheduler)
//! can invoke the algorithm at a later point to continue its work.

use super::graph::Node;
use crate::signal_handling::received_ctrl_c;
use std::time::{Duration, Instant};

/// [`IterativeAlgorithm`] provides a consistent interface to execute all our algorithms. Observe
/// that it does not prescribe any constructor which is left to the algorithm designer as each
/// algorithm has specific parameters et cetera. The construction phase should, in general, be
/// quite fast and only involve little computation.
///
/// As an adopter of [`IterativeAlgorithm`], you have to implement at least the methods
///   [`IterativeAlgorithm::execute_step`],
///   [`IterativeAlgorithm::is_completed`] and [`IterativeAlgorithm::best_known_solution`].
/// You may also implement [`IterativeAlgorithm::best_known_solution_size`] if the size of the solution
/// can be computed more cheaply than the solution itself.
///
/// If your algorithm is known to eventually terminate please also implement the marker trait
/// [`TerminatingIterativeAlgorithm`]. It offers and easy interface to run the algorithm to completion.
///
/// # Example
/// ```
/// use dfvs::algorithm::IterativeAlgorithm;
/// use dfvs::graph::{AdjacencyList, Node};
/// struct MyAlgorithm<'a, G> {
///    graph: &'a G,
///    solution: Option<Vec<Node>>
/// }
///
/// impl<'a, G> IterativeAlgorithm for MyAlgorithm<'a, G> where G: 'a + AdjacencyList {
///     fn execute_step(&mut self) {
///         // do some magic to improve the solution. To avoid overhead, the computation
///         // should take at least a few milliseconds but less than a second.
///     }
///
///     fn is_completed(&self) -> bool {
///         // do some magic to decide whether the solution is already optimal
///         false
///     }
///
///     fn best_known_solution(&mut self) -> Option<&[Node]> {
///         self.solution.as_deref()
///     }
/// }
/// ```
pub trait IterativeAlgorithm {
    /// Advances the computation of this algorithm. The execution should take between on the order of
    /// several milliseconds and not significantly exceed a second for expected inputs.
    fn execute_step(&mut self);

    /// Returns true iff the algorithm is completed and [`IterativeAlgorithm::execute_step`] may not
    /// be called again.
    fn is_completed(&self) -> bool;

    /// Returns the currently best known solution or None if no solution is known yet.
    fn best_known_solution(&mut self) -> Option<&[Node]>;

    /// Returns the size of the currently best known solution.
    fn best_known_solution_size(&mut self) -> Option<Node> {
        Some(self.best_known_solution()?.len() as Node)
    }

    /// Execute the algorithm and keeps calling [`IterativeAlgorithm::execute_step`] until the
    /// [`predicate`] becomes false, a termination signal was received, or [`IterativeAlgorithm::is_completed`]
    /// becomes true. The function `predicate` is evaluated after each iteration, i.e. a step is
    /// carried out even if the predicate always returns false.
    fn run_while<F: Fn(&mut Self) -> bool>(&mut self, predicate: F) {
        while !self.is_completed() && !received_ctrl_c() {
            self.execute_step();

            if predicate(self) {
                break;
            }
        }
    }

    /// Execute the algorithm and keeps calling [`IterativeAlgorithm::execute_step`] until either a
    /// timeout occurred, a termination signal was received, or [`IterativeAlgorithm::is_completed`]
    /// is true. Observe that the timeout is guaranteed only in the sense that
    /// [`IterativeAlgorithm::execute_step`] is not called again after the timeout; if the function
    /// should take too long (or not return at all) the timeout will be violated.
    fn run_until_timeout(&mut self, timeout: Duration) {
        let start = Instant::now();
        self.run_while(|_| start.elapsed() < timeout);
    }
}

/// [`TerminatingIterativeAlgorithm`] is a marker trait, i.e. to adopt it, you give an empty `impl`
/// block. Add this trait to algorithms that will eventually terminate (i.e. in contrast to an
/// algorithm does not know when to stop).
///
/// # Example
///
/// ```ignore
/// use dfvs::algorithm::{IterativeAlgorithm, TerminatingIterativeAlgorithm};
///
/// struct MyAlgorithm {};
///
/// impl IterativeAlgorithm for MyAlgorithm {
///    /* implement all methods required */
/// }
///
/// impl TerminatingIterativeAlgorithm for MyAlgorithm {
///    // no implementation required
/// }
/// ```
pub trait TerminatingIterativeAlgorithm: IterativeAlgorithm {
    /// Execute the algorithm until it completed (or the termination signal was received) and
    /// return the solution if it was found.
    fn run_to_completion(&mut self) -> Option<Vec<Node>> {
        while !self.is_completed() && !received_ctrl_c() {
            self.execute_step();
        }
        Some(self.best_known_solution()?.to_vec())
    }
}
