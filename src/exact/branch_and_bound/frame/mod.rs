use super::*;
use crate::heuristics::lowerbound_circuits::LowerBound;
use crate::kernelization::*;
use fxhash::FxHashSet;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use std::iter::FromIterator;
use std::time::Duration;

mod branch_on_clique;
mod branch_on_node;
mod branch_on_node_group;
mod branch_on_sccs;
mod cuts;
mod kernelization;

const MIN_REDUCED_SCC_SIZE: Node = 2;
const TRIVIAL_KERNEL_RULES_ONLY: bool = false;
const USE_ADAPTIVE_KERNEL_FREQUENCY: bool = false;
const DELETE_TWINS_MIRRORS_AND_SATELLITES: bool = true;
const MATRIX_SOLVER_SIZE: Node = 64;
const SEARCH_CUTS: bool = true;
const BRANCH_ON_CYCLES: bool = true;
const BRANCH_ON_CLIQUES: bool = true;
const ALWAYS_COMPUTE_LOWER_BOUNDS: bool = false;

/// The frame implements the heavy lifting of the branch and bound algorithm (see also [`BranchAndBound`]
/// for further information). After construction, the computation starts with [`Frame::initialize`].
/// It may either directly return a result using [`BBResult::Result`] or branch by returning [`BBResult::Branch`].
/// In the latter case, we expect the child's computation to complete before calling the parents [`Frame::resume`]
/// method. Internally, the resumption point (and information to resume) are kept in the attribute [`Frame::resume_with`].
pub(super) struct Frame<G> {
    /// The graph instance to be solved. It may get updated (e.g. by kernelization) or even taken (e.g. to pass to a
    /// child without copying if we know that the graph is not required by the parent anymore).
    pub(super) graph: G,

    /// upper bound on the solution size for `graph` excluding the node in `partial_solution`.
    /// The upper bound is EXCLUSIVE, i.e. a solution must always have size `solution.len() < upper_bound as usize`.
    pub(super) upper_bound: Node,

    /// Lower bound on the solution size for `graph` excluding the node in `partial_solution`.
    /// The lower bound is inclusive, i.e. a solution has always size `solution.len() >= lower_bound as usize`.
    pub(super) lower_bound: Node,

    pub(super) resume_with: FrameState<G>,

    /// The partial result obtained so far is stored in `partial_solution`.  We may need to build a
    /// solution in several steps (e.g. during kernelization, when splitting SCCs, et cetera). If we
    /// add to `partial_solution` we need to decrement the lower- and upper-bound accordingly.
    /// Note that a parent may use `partial_solution` to remember node deletions. So do not assume that
    /// `partial_solution` is empty before calling [`Frame::initialize`]
    pub(super) partial_solution: Solution,
    pub(super) partial_solution_parent: Solution,

    /// Optimization: we safe the size of the largest clique found during lower bound computation and
    /// pass this information along to our children to accelerate their lower bound computation.
    pub(super) max_clique: Option<Node>,

    /// Optimization: we may skip certain steps (e.g., computation of SCCs) or active certain
    /// branching rules if we know the graph to be connected.
    pub(super) graph_is_connected: Option<bool>,

    /// Optimization: a parent may inform its children that the graph is known to be reduced and
    /// no kernelization is necessary
    pub(super) graph_is_reduced: bool,

    /// Postprocessor that keeps the state of two-staged kernelization rules. It is used in the
    /// terminating methods [`Frame::add_partial_solution`] and [`Frame::partial_solution_as_result`]
    pub(super) postprocessor: Postprocessor,

    pub(super) directed_mode: bool,

    pub(super) initial_upper_bound: Node,
    pub(super) initial_lower_bound: Node,

    graph_digest: String,
    kernel_misses: KernelMisses,
}

#[derive(Debug)]
pub(super) enum FrameState<G> {
    Uninitialized,
    ResumeDeleteBranch(Node, Node),
    ResumeContractBranch(/* from_delete: */ OptSolution),
    ResumeSCCSplit(Vec<(G, NodeMapper)>, Vec<Node>),
    ResumeNodeGroup(OptSolution, Vec<Vec<Node>>, Vec<Node>, Vec<Node>),
    ResumeClique(OptSolution, Vec<Node>, Node),
    ResumeMapped(NodeMapper),
}

impl<G> FrameState<G> {
    pub(super) fn describe(&self) -> String {
        match self {
            Uninitialized => String::from("Uninitialized"),
            ResumeDeleteBranch(_, num) => format!("DeleteBranch(deleted={})", num),
            ResumeContractBranch(_) => String::from("ContractBranch"),
            ResumeSCCSplit(branches, _) => format!("SCCSplit(remaining: {})", branches.len() - 1),
            ResumeNodeGroup(_, _, _, _) => String::from("ResumeNodeGroup"),
            ResumeClique(_, _, _) => String::from("Clique"),
            ResumeMapped(_) => String::from("Mapped"),
        }
    }
}

#[derive(Default, Clone, Debug, Eq, PartialEq)]
struct KernelMisses {
    di_cliques: Node,
    pie: Node,
    c4: Node,
    undir_dom_node: Node,
    domn: Node,
    redundant_cycle: Node,
    unconfined: Node,
    crown: Node,
    dom_node_edges: Node,
    rules56: Node,
    undir_degree_two: Node,
    funnel: Node,
    twins_degree_three: Node,
    desk: Node,
}

use FrameState::*;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn new(graph: G, lower_bound: Node, upper_bound: Node) -> Self {
        Self {
            graph,
            resume_with: Uninitialized,
            upper_bound,
            initial_upper_bound: upper_bound,
            lower_bound,
            initial_lower_bound: lower_bound,
            max_clique: None,
            graph_is_connected: None,
            graph_is_reduced: false,
            partial_solution: Vec::new(),
            partial_solution_parent: Vec::new(),
            postprocessor: Postprocessor::new(),
            directed_mode: false,
            graph_digest: String::new(),
            kernel_misses: Default::default(),
        }
    }

    pub(super) fn graph_digest(&mut self) -> &String {
        if self.graph_digest.is_empty() {
            self.graph_digest = self.graph.digest_sha256();
        }

        &self.graph_digest
    }

    /// This function carries out kernelization and decides upon a branching strategy.
    pub(super) fn initialize(&mut self) -> BBResult<G> {
        assert!(self.partial_solution.is_empty());
        assert!(self
            .partial_solution_parent
            .iter()
            .all(|&u| self.graph.total_degree(u) == 0));

        assert_eq!(self.upper_bound, self.initial_upper_bound);
        assert_eq!(self.lower_bound, self.initial_lower_bound);

        macro_rules! try_branch {
            ($r:expr) => {
                if let Some(b) = $r {
                    return b;
                }
            };
        }

        if self.lower_bound >= self.upper_bound {
            return self.fail();
        }

        if self.graph.number_of_edges() == 0 {
            return self.return_partial_solution_as_result();
        }

        // kernelization
        try_branch!(self.apply_kernelization());

        // process small instances with matrix-solver
        #[allow(clippy::absurd_extreme_comparisons)]
        if self.graph.number_of_nodes() < MATRIX_SOLVER_SIZE {
            let sol = branch_and_bound_matrix(&self.graph, Some(self.upper_bound - 1));
            return self.return_to_result_and_partial_solution(sol);
        }

        try_branch!(self.try_branch_on_sccs());

        if !self.directed_mode {
            try_branch!(self.try_branch_on_simple_cuts());
        }

        let mut cycle = None;

        let clique = if self.max_clique.map_or(true, |c| c > 4) {
            let max_size = self.max_clique.unwrap_or(30);
            let (clique, timeout) = self.graph.maximum_complete_subgraph_with_timeout(
                Some(max_size),
                Some(Duration::from_millis(100)),
            );
            if let Some(nodes) = clique.as_ref() {
                if !timeout {
                    self.max_clique = Some(nodes.len() as Node);
                    if nodes.len() > 5 {
                        return self.branch_on_clique(nodes.clone());
                    }
                }
            }
            clique
        } else {
            None
        };

        #[allow(clippy::absurd_extreme_comparisons)]
        if clique.is_none()
            && (ALWAYS_COMPUTE_LOWER_BOUNDS
                || self.lower_bound <= MIN_REDUCED_SCC_SIZE
                || self.lower_bound * 4 > self.upper_bound * 3
                || (!self.directed_mode && self.max_clique.map_or(true, |c| c > 3))
                || thread_rng().gen_ratio(1, 16))
        {
            let mut lb = LowerBound::new(&self.graph);
            if let Some(mc) = self.max_clique {
                lb.set_max_clique(mc);
            }
            lb.compute();
            self.max_clique = Some(lb.largest_clique().map_or(0, |c| c.len() as Node));
            self.lower_bound = self
                .lower_bound
                .max(lb.lower_bound())
                .max(MIN_REDUCED_SCC_SIZE);
            if self.upper_bound <= self.lower_bound {
                return self.fail();
            }

            cycle = lb.shortest_cycle().map(Vec::from);

            if let Some(clique) = lb.largest_clique() {
                if BRANCH_ON_CLIQUES && clique.len() >= 3 {
                    return self.branch_on_clique(Vec::from(clique));
                }
            }
        }

        if SEARCH_CUTS {
            try_branch!(self.try_branch_on_fancy_cuts());
        }

        if let Some(clique) = clique {
            if BRANCH_ON_CLIQUES && clique.len() >= 3 {
                return self.branch_on_clique(clique);
            }
        }

        // try to branch on the shortest cycle found during lower bound computation
        if let Some(cycle) = cycle {
            if BRANCH_ON_CYCLES && self.directed_mode && cycle.len() > 2 {
                return self.branch_on_node_group(cycle);
            }
        }

        // if no other branching method worked, branch on the fittest remaining node (according to `branching_score`)
        self.branch_on_node(
            self.graph
                .vertices()
                .max_by_key(|&u| self.branching_score(u))
                .unwrap(),
        )
    }

    /// This function is called if this previously branched using `BBResult::Branch` and the computation
    /// at the child (and possibly its children) completed. We then receive the result computed by
    /// the child.
    pub(super) fn resume(&mut self, from_child: OptSolution) -> BBResult<G> {
        let mut state = Uninitialized;
        std::mem::swap(&mut self.resume_with, &mut state);

        match state {
            ResumeDeleteBranch(node, num_nodes_deleted) => {
                self.resume_delete_branch(node, num_nodes_deleted, Some(from_child))
            }

            ResumeContractBranch(from_delete) => {
                self.resume_contract_branch(from_delete, from_child)
            }
            ResumeSCCSplit(subgraphs, lower_bounds) => {
                self.resume_scc_split(subgraphs, lower_bounds, Some(from_child))
            }

            ResumeNodeGroup(current_best, branches, cut, cut_from_child) => self.resume_node_group(
                current_best,
                branches,
                cut,
                Some(cut_from_child),
                from_child,
            ),

            ResumeClique(current_best, clique, i) => {
                self.resume_clique(current_best, clique, i, from_child)
            }

            ResumeMapped(mapper) => self.resume_mapped(mapper, from_child),

            Uninitialized => panic!("Resume uninitialized frame"),
        }
    }

    /// Create a new instance of Self and copies some of the parent's values
    fn branch(&self, graph: G, lower_bound: Node, upper_bound: Node) -> Self {
        let mut branch = Self::new(graph, lower_bound, upper_bound);
        branch.max_clique = self.max_clique;
        branch.directed_mode = self.directed_mode;
        branch
    }

    /// Computes mirror nodes as described proposed by Fomin et al. in "A measure & conquer approach
    /// for the analysis of exact algorithms". If we decide to put a node u into the solution, the
    /// set of mirrors needs to follow.
    fn get_mirrors(&self, graph: &G, u: Node) -> Vec<Node> {
        if !DELETE_TWINS_MIRRORS_AND_SATELLITES {
            unreachable!();
        }

        fn is_di_clique<G: AdjacencyListUndir, I: Iterator<Item = Node>>(
            graph: &G,
            node: Option<Node>,
            neighbors: I,
        ) -> bool {
            let nodes = FxHashSet::from_iter(neighbors);
            if nodes.is_empty() {
                return true;
            }

            let k = nodes.len() - 1 + (node.is_some() as usize);

            nodes.iter().all(|&v| graph.undir_degree(v) >= k as Node)
                && nodes.iter().all(|&v| {
                    graph
                        .undir_neighbors(v)
                        .filter(|&x| Some(x) == node || nodes.contains(&x))
                        .take(k)
                        .count()
                        == k
                })
        }

        let mut two_neighbors = BitSet::new_all_unset_but(
            graph.len(),
            graph
                .undir_neighbors(u)
                .flat_map(|v| graph.undir_neighbors(v))
                .map(|v| v as usize),
        );

        let nu = BitSet::new_all_unset_but(
            graph.len(),
            graph
                .out_neighbors(u)
                .chain(graph.in_neighbors(u))
                .map(|v| v as usize),
        );
        two_neighbors.and_not(&nu);
        two_neighbors.unset_bit(u as usize);

        two_neighbors
            .iter()
            .map(|v| v as Node)
            .filter(|&v| {
                let mut nu_without_nv = nu.clone();
                for x in graph.undir_neighbors(v) {
                    nu_without_nv.unset_bit(x as usize);
                }

                nu_without_nv.is_empty()
                    || is_di_clique(graph, None, nu_without_nv.iter().map(|x| x as Node))
            })
            .collect_vec()
    }

    /// Computes satellites nodes as described proposed by Fomin et al. in "A measure & conquer approach
    /// for the analysis of exact algorithms". If we decide to put a node u into the solution, the
    /// set of mirrors needs to follow.
    fn get_satellite(&self, graph: &G, node: Node) -> Vec<Node> {
        if !DELETE_TWINS_MIRRORS_AND_SATELLITES {
            unreachable!();
        }

        let undir_neighbors = graph
            .undir_neighbors(node)
            .chain(std::iter::once(node))
            .collect_vec();

        let mut sats = graph
            .undir_neighbors(node)
            .filter_map(|u| {
                if graph.has_dir_edges(u) || graph.undir_degree(u) > undir_neighbors.len() as Node {
                    return None;
                }

                let s = graph
                    .undir_neighbors(u)
                    .filter(|v| !undir_neighbors.contains(v))
                    .take(2)
                    .collect_vec();

                if s.len() == 1 {
                    Some(s[0])
                } else {
                    None
                }
            })
            .collect_vec();

        sats.sort_unstable();
        sats.dedup();
        sats
    }

    /// Computes some fitness heuristic to select a good candidate to branch on. The higher the
    /// score, the better to branch on it.
    fn branching_score(&self, u: Node) -> usize {
        // the assumption for this heuristic: undirected edges are more valuable than directed ones;
        // so weight them accordingly
        self.graph.len() * self.graph.undir_degree(u) as usize
            + self.graph.in_degree(u) as usize * self.graph.out_degree(u) as usize
    }

    /// Shortcut to indicate the the computation of this branch failed. Use as `return self.fail();`
    fn fail(&self) -> BBResult<G> {
        BBResult::Result(None)
    }

    /// Shortcut to indicate that `self.partial_solution` contains the minimal solution of our frame
    /// (post-processing may still be necessary and is carried out by the method).  Use as
    /// `return self.return_partial_solution_as_result()`
    fn return_partial_solution_as_result(&mut self) -> BBResult<G> {
        let mut res = std::mem::take(&mut self.partial_solution);
        res.extend(&self.partial_solution_parent);
        self.postprocessor.finalize_with_global_solution(&mut res);
        BBResult::Result(Some(res))
    }

    /// Shortcut to indicate that `solution` (and possibly `self.partial_solution`) contains the
    /// minimal solution of our frame (post-processing may still be necessary and is carried out by the
    /// method). If `solution` is `None` it indicates that no feasible solution could be obtained,
    /// If it is `Some(sol)` the solution return consists of the post-processed union of `sol` and
    /// `self.partial_solution`. Use as
    /// `return self.return_partial_solution_as_result(solution)`
    fn return_to_result_and_partial_solution(&mut self, solution: OptSolution) -> BBResult<G> {
        BBResult::Result(solution.map(|mut sol| {
            sol.extend(&self.partial_solution);
            sol.extend(&self.partial_solution_parent);
            self.postprocessor.finalize_with_global_solution(&mut sol);
            sol
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glob::glob;
    use std::str::FromStr;

    pub(super) fn for_each_stress_graph<F: Fn(&String, &AdjArrayUndir) -> ()>(callback: F) {
        glob("data/stress_kernels/*_n*_m*_0[01]*.metis")
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .iter()
            .filter_map(|filename| {
                let file_extension = filename.extension().unwrap().to_str().unwrap().to_owned();
                let file_format = FileFormat::from_str(&file_extension).ok()?;
                let graph = AdjArrayUndir::try_read_graph(file_format, filename).ok()?;
                if graph.len() < 90 {
                    Some((String::from(filename.to_str().unwrap()), graph))
                } else {
                    None
                }
            })
            .for_each(|(file, graph)| callback(&file, &graph));
    }

    pub(super) fn for_each_stress_graph_with_opt_sol<F: Fn(&String, &AdjArrayUndir, Node) -> ()>(
        callback: F,
    ) {
        for_each_stress_graph(|filename, graph| {
            let opt_sol_size = branch_and_bound_matrix(graph, None).unwrap().len() as Node;
            callback(filename, graph, opt_sol_size)
        })
    }

    /// Simulate the execution of a branched frame. Each call to a child is processed using the
    /// matrix solver; thus it helps in build unit tests.
    pub(super) fn simulate_execution(
        frame: &mut Frame<AdjArrayUndir>,
        first_result: BBResult<AdjArrayUndir>,
    ) -> OptSolution {
        let mut result = first_result;
        loop {
            match result {
                BBResult::Result(res) => return res,
                BBResult::Branch(child) => {
                    let response = if child.upper_bound == 0 {
                        None
                    } else {
                        let mut sol = branch_and_bound_matrix(&child.graph, None).unwrap();
                        assert!(child.lower_bound <= sol.len() as Node);
                        if child.upper_bound > sol.len() as Node {
                            sol.extend(&child.partial_solution_parent);
                            Some(sol)
                        } else {
                            None
                        }
                    };

                    if false {
                        let parent_len = child.partial_solution_parent.len() as Node;
                        println!("Simulated child (lb={}|{}, ub={}|{}, parent={:?}) and return: {:?} (k={})",
                                 child.initial_lower_bound,
                                 child.initial_lower_bound + parent_len,
                                 child.initial_upper_bound,
                                 child.initial_upper_bound + parent_len,
                                 &child.partial_solution_parent,
                                 &response, response.as_ref().map_or(-1, |s| s.len() as isize));
                    }

                    result = frame.resume(response);
                }
            }
        }
    }

    fn integration_stress_test<FB: Fn(AdjArrayUndir, Node) -> Frame<AdjArrayUndir>>(
        frame_builder: FB,
        expect_success: bool,
    ) {
        for_each_stress_graph_with_opt_sol(|filename, graph, opt_sol| {
            // we high-jack the frame_builder idiom from the actual branching tests and build an
            // algorithm instance from it; not very clean, but it works
            let mut frame = frame_builder(graph.clone(), opt_sol);
            let mut algo = BranchAndBound::new(std::mem::take(&mut frame.graph));
            algo.set_lower_bound(frame.lower_bound);
            algo.set_upper_bound(frame.upper_bound.saturating_sub(1));

            let result = algo.run_to_completion();

            if expect_success {
                let my_size = result.as_ref().map_or(-1, |s| s.len() as isize);

                assert_eq!(
                    my_size, opt_sol as isize,
                    "file: {} opt: {} my-size: {} my-sol: {:?}",
                    filename, opt_sol, my_size, result
                );
            } else {
                assert!(result.is_none())
            }
        });
    }

    branching_stress_tests!(integration_stress_test);

    macro_rules! branching_stress_tests {
        ($st:ident) => {
            #[test]
            fn stress_relaxed_lower_upper() {
                $st(
                    |graph, _| Frame::new(graph.clone(), 0, graph.number_of_nodes()),
                    true,
                );
            }

            #[test]
            fn stress_too_low_upper() {
                $st(
                    |graph, opt_sol| Frame::new(graph.clone(), 0, opt_sol),
                    false,
                );
            }

            #[test]
            fn stress_tight_upper() {
                $st(
                    |graph, opt_sol| Frame::new(graph.clone(), 0, opt_sol + 1),
                    true,
                );
            }

            #[test]
            fn stress_tight_lower() {
                $st(
                    |graph, opt_sol| Frame::new(graph.clone(), opt_sol, graph.number_of_nodes()),
                    true,
                );
            }

            #[test]
            fn stress_lower_and_upper() {
                $st(
                    |graph, opt_sol| Frame::new(graph.clone(), opt_sol, opt_sol + 1),
                    true,
                );
            }
        };
    }

    pub(super) use branching_stress_tests;
}
