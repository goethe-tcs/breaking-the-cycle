use super::cuts::*;
use super::*;
use crate::heuristics::lowerbound_circuits::LowerBound;
use crate::kernelization::*;
use crate::utils::*;
use fxhash::FxHashSet;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use std::iter::FromIterator;

#[derive(Debug)]
pub(super) enum FrameState<G> {
    Uninitialized,
    ResumeDeleteBranch(Node, Node),
    ResumeContractBranch(/* from_delete: */ OptSolution),
    ResumeSCCSplit(Vec<(G, NodeMapper)>, Vec<Node>),
    ResumeVertexCut(OptSolution, Vec<Node>, AllIntSubsets, bool),
    ResumeClique(OptSolution, Vec<Node>, Node),
    ResumeMapped(NodeMapper),
}

use FrameState::*;

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
}

const MIN_REDUCED_SCC_SIZE: Node = 2;
const DELETE_TWINS_MIRRORS_AND_SATELLITES: bool = true;
const MATRIX_SOLVER_SIZE: Node = 64;
const SEARCH_CUTS: bool = true;
const DELETE_CYCLES: bool = true;
const DELETE_CLIQUES: bool = true;
const ALWAYS_COMPUTE_LOWER_BOUNDS: bool = false;

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
        }
    }

    pub(super) fn graph_digest(&mut self) -> &String {
        if self.graph_digest.is_empty() {
            self.graph_digest = self.graph.digest_sha256();
        }

        &self.graph_digest
    }

    fn branch(&self, graph: G, lower_bound: Node, upper_bound: Node) -> Self {
        let mut branch = Self::new(graph, lower_bound, upper_bound);
        branch.max_clique = self.max_clique;
        branch
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

        if self.lower_bound >= self.upper_bound {
            return self.fail();
        }

        if self.graph.number_of_edges() == 0 {
            return self.return_partial_solution_as_result();
        }

        // kernelization
        if let Some(result) = self.apply_kernelization() {
            return result;
        }

        if self.lower_bound >= self.upper_bound {
            return self.fail();
        }

        // process small instances with matrix-solver
        #[allow(clippy::absurd_extreme_comparisons)]
        if self.graph.number_of_nodes() < MATRIX_SOLVER_SIZE {
            return self.return_to_result_and_partial_solution(
                crate::exact::branch_and_bound_matrix::branch_and_bound_matrix(
                    &self.graph,
                    Some(self.upper_bound.saturating_sub(1)),
                ),
            );
        }

        if let Some(result) = self.try_split_into_sccs() {
            return result;
        }

        let mut cycle = None;

        #[allow(clippy::absurd_extreme_comparisons)]
        if ALWAYS_COMPUTE_LOWER_BOUNDS
            || self.lower_bound <= MIN_REDUCED_SCC_SIZE
            || self.lower_bound * 4 > self.upper_bound * 3
            || (!self.directed_mode && self.max_clique.map_or(true, |c| c > 3))
            || thread_rng().gen_ratio(1, 16)
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
                let mut clique = Vec::from(clique);

                if DELETE_CLIQUES && clique.len() >= 3 {
                    clique.sort_unstable_by_key(|&u| -(self.graph.total_degree(u) as i64));
                    return self.resume_clique(None, clique, 0, None);
                }
            }
        }

        if SEARCH_CUTS {
            if let Some(result) = self.try_branching_on_fancy_cuts() {
                return result;
            }
        }

        // try to branch on the shortest cycle found during lower bound computation
        if let Some(cycle) = cycle {
            if DELETE_CYCLES && self.directed_mode && cycle.len() > 2 {
                let k = cycle.len();
                return self.resume_vertex_cut(
                    None,
                    cycle,
                    AllIntSubsets::new(k as u32),
                    false,
                    None,
                );
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

            ResumeVertexCut(current_best, cut, subset_iter, reversed) => {
                self.resume_vertex_cut(current_best, cut, subset_iter, reversed, from_child)
            }

            ResumeClique(current_best, clique, i) => {
                self.resume_clique(current_best, clique, i, from_child)
            }

            ResumeMapped(mapper) => self.resume_mapped(mapper, from_child),

            Uninitialized => panic!("Resume uninitialized frame"),
        }
    }

    fn resume_clique(
        &mut self,
        mut current_best: OptSolution,
        clique: Vec<Node>,
        i: Node,
        from_child: OptSolution,
    ) -> BBResult<G> {
        if from_child.is_some() {
            current_best = from_child;
            self.upper_bound = current_best.as_ref().unwrap().len() as Node;
            if self.upper_bound == self.lower_bound {
                return self.return_to_result_and_partial_solution(current_best);
            }
        }

        if i == clique.len() as Node + 1 {
            return self.return_to_result_and_partial_solution(current_best);
        }

        let mut subgraph = if i == clique.len() as Node {
            std::mem::take(&mut self.graph) // last iteration --- do not need the original graph anymore
        } else {
            self.graph.clone()
        };

        let (nodes_deleted, node_to_spare) = if i == 0 {
            // delete whole clique
            (clique.clone(), None)
        } else {
            let mut nodes = clique.clone();
            nodes.swap(i as usize - 1, clique.len() - 1);
            let spare = nodes.pop().unwrap();

            (nodes, Some(spare))
        };

        let mut into_solution = nodes_deleted.clone();

        for &v in &nodes_deleted {
            into_solution.extend(self.remove_node_and_dominating(&mut subgraph, v));
        }

        into_solution.sort_unstable();
        into_solution.dedup();

        if let Some(node) = node_to_spare {
            if into_solution.contains(&node) {
                return self.resume_clique(current_best, clique, i + 1, None);
            }

            // should not happen since reduction will remove self-loops
            debug_assert!(!subgraph.has_edge(node, node));
            subgraph.contract_node(node);
        }

        let num_nodes_deleted = into_solution.len() as Node;
        if num_nodes_deleted >= self.upper_bound {
            return self.resume_clique(current_best, clique, i + 1, None);
        }

        let mut branch = self.branch(
            subgraph,
            self.lower_bound.saturating_sub(num_nodes_deleted),
            self.upper_bound - num_nodes_deleted,
        );
        assert_eq!(self.max_clique, Some(clique.len() as Node));
        branch.max_clique = self.max_clique;
        branch.partial_solution_parent = into_solution;

        self.resume_with = ResumeClique(current_best, clique, i + 1);
        BBResult::Branch(branch)
    }

    fn resume_vertex_cut(
        &mut self,
        mut current_best: OptSolution,
        cut: Vec<Node>,
        mut subset_iter: AllIntSubsets,
        reversed: bool,
        from_child: OptSolution,
    ) -> BBResult<G> {
        assert!(!cut.is_empty());
        assert!(cut
            .iter()
            .all(|&u| self.graph.in_degree(u) > 0 && self.graph.out_degree(u) > 0));

        if let Some(sol) = from_child {
            debug_assert!(sol.len() < self.upper_bound as usize);
            self.upper_bound = sol.len() as Node;
            current_best = Some(sol);
            if self.upper_bound == self.lower_bound {
                return self.return_to_result_and_partial_solution(current_best);
            }
        }

        // more recursions to follow
        'mask: for mask in subset_iter.by_ref() {
            let mask = if reversed {
                !mask & u64::lowest_bits_set(cut.len()) // we start by including all nodes first
            } else {
                mask
            };
            let nodes_deleted = mask.count_ones();

            if nodes_deleted >= self.upper_bound {
                continue;
            }

            let mut branch = Self::new(
                self.graph.clone(),
                self.lower_bound, // we be reduced later
                self.upper_bound,
            );

            branch.partial_solution_parent.reserve(nodes_deleted as usize);
            branch.max_clique = self.max_clique;

            for (_, &node) in cut
                .iter()
                .enumerate()
                .filter(|(i, _)| mask.is_ith_bit_set(*i))
            {
                if branch.graph.in_degree(node) == 0 || branch.graph.out_degree(node) == 0 {
                    continue 'mask;
                }

                branch
                    .partial_solution_parent
                    .extend(self.remove_node_and_dominating(&mut branch.graph, node));
                branch.partial_solution_parent.push(node);
            }

            for (_, &node) in cut
                .iter()
                .enumerate()
                .filter(|(i, _)| !mask.is_ith_bit_set(*i))
            {
                if branch.graph.has_edge(node, node) {
                    continue 'mask; // we cannot contract a node with a self-loop (which might have
                                    // been introduced by an earlier contraction after reduction)
                }

                let deleted = branch.graph.contract_node(node);
                branch.partial_solution_parent.extend(&deleted);
                for d in deleted {
                    branch
                        .partial_solution_parent
                        .extend(self.remove_node_and_dominating(&mut branch.graph, d));
                }
            }

            if branch.partial_solution_parent.len() > nodes_deleted as usize {
                if branch.partial_solution_parent.len() >= self.upper_bound as usize {
                    continue;
                }

                branch.partial_solution_parent.sort_unstable();
                branch.partial_solution_parent.dedup();
            }

            branch.graph_is_connected = if branch.partial_solution_parent.is_empty() {
                self.graph_is_connected
            } else {
                None
            };

            if branch.upper_bound <= branch.partial_solution_parent.len() as Node {
                continue;
            }

            branch.upper_bound -= branch.partial_solution_parent.len() as Node;
            branch.lower_bound = branch
                .lower_bound
                .saturating_sub(branch.partial_solution_parent.len() as Node);

            branch.initial_upper_bound = branch.upper_bound;
            branch.initial_lower_bound = branch.lower_bound;

            self.resume_with = ResumeVertexCut(current_best, cut, subset_iter, reversed);

            return BBResult::Branch(branch);
        }

        self.return_to_result_and_partial_solution(current_best)
    }

    fn resume_scc_split(
        &mut self,
        mut subgraphs: Vec<(G, NodeMapper)>,
        mut lower_bounds: Vec<Node>,
        from_child: Option<OptSolution>,
    ) -> BBResult<G> {
        if let Some(opt_sol) = from_child {
            if let Some(sol) = opt_sol {
                self.upper_bound -= sol.len() as Node;

                let mapper = subgraphs.pop().unwrap().1;
                self.partial_solution
                    .extend(sol.into_iter().map(|u| mapper.old_id_of(u).unwrap()));

                debug_assert_eq!(subgraphs.len(), lower_bounds.len());
            } else {
                return self.fail();
            }
        }

        if let Some(next_branch) = subgraphs.last_mut() {
            let lower_bound = lower_bounds.pop().unwrap();
            let remaining_lower = lower_bounds.iter().sum::<Node>();

            if remaining_lower + lower_bound >= self.upper_bound {
                return self.fail();
            }

            let scc = std::mem::take(&mut next_branch.0);

            let upper_bound = (self.upper_bound - remaining_lower).min(scc.number_of_nodes());

            let mut branch = self.branch(scc, lower_bound, upper_bound);
            branch.graph_is_connected = Some(true);

            self.resume_with = ResumeSCCSplit(subgraphs, lower_bounds);

            return BBResult::Branch(branch);
        }

        self.return_partial_solution_as_result()
    }

    fn branch_on_node(&mut self, node: Node) -> BBResult<G> {
        assert!(self.graph.in_degree(node) > 0 && self.graph.out_degree(node) > 0);

        let mut graph = self.graph.clone();
        let mut partial_solution = self.remove_node_and_dominating(&mut graph, node);
        assert!(!partial_solution.contains(&node));
        partial_solution.push(node);

        let num_nodes_deleted = partial_solution.len() as Node;

        if num_nodes_deleted >= self.upper_bound {
            return self.resume_delete_branch(node, num_nodes_deleted, None);
        }

        self.resume_with = ResumeDeleteBranch(node, num_nodes_deleted);

        let mut branch = self.branch(
            graph,
            self.lower_bound.saturating_sub(num_nodes_deleted),
            self.upper_bound - num_nodes_deleted,
        );
        branch.partial_solution_parent = partial_solution;

        BBResult::Branch(branch)
    }

    fn resume_delete_branch(
        &mut self,
        node: Node,
        _num_nodes_deleted: Node,
        mut from_child: Option<OptSolution>,
    ) -> BBResult<G> {
        // Add the previously deleted node
        if let Some(from_child) = from_child.as_mut() {
            if let Some(sol) = from_child.as_mut() {
                debug_assert!(self.upper_bound > sol.len() as Node);
                self.upper_bound = sol.len() as Node;

                if self.upper_bound <= self.lower_bound {
                    return self.return_to_result_and_partial_solution(Some(sol.clone()));
                }
            }
        }

        if self.graph.has_edge(node, node) || self.graph.undir_degree(node) >= self.upper_bound {
            return self.return_to_result_and_partial_solution(from_child.unwrap_or(None));
        }

        // Setup re-entry
        self.resume_with = ResumeContractBranch(from_child.unwrap_or(None));

        // Setup child call
        let mut subgraph = std::mem::take(&mut self.graph);
        let removed = subgraph.contract_node(node);
        subgraph.remove_edges_of_nodes(&removed);

        let mut branch = self.branch(
            subgraph,
            self.lower_bound.saturating_sub(removed.len() as Node),
            self.upper_bound - removed.len() as Node,
        );
        branch.graph_is_connected = if removed.is_empty() {
            self.graph_is_connected
        } else {
            None
        };
        branch.max_clique = self.max_clique.map(|mc| mc + 1);
        branch.partial_solution_parent = removed;
        BBResult::Branch(branch)
    }

    fn resume_contract_branch(
        &mut self,
        from_delete: OptSolution,
        from_child: OptSolution,
    ) -> BBResult<G> {
        self.return_to_result_and_partial_solution(from_child.or(from_delete))
    }

    fn resume_mapped(&mut self, mapper: NodeMapper, mut from_child: OptSolution) -> BBResult<G> {
        if let Some(sol) = from_child.as_mut() {
            for u in sol {
                *u = mapper.old_id_of(*u).unwrap();
            }
        }

        self.return_to_result_and_partial_solution(from_child)
    }

    /// If we decide to put a node u into the solution, we can remove it from the graph. Also there
    /// are dominated vertices (which have a subset of in- and out-neighbors of u) as well as mirrors
    /// (see [`Frame::get_mirrors`]) that can also be removed.
    fn remove_node_and_dominating(&self, graph: &mut G, node: Node) -> Vec<Node> {
        if !DELETE_TWINS_MIRRORS_AND_SATELLITES {
            graph.remove_edges_at_node(node);
            return Vec::new();
        }

        if let Some(pred) = graph
            .in_neighbors(node)
            .min_by_key(|&p| graph.out_degree(p))
        {
            let mut preds = FxHashSet::from_iter(graph.in_neighbors(node));
            let mut succ = FxHashSet::from_iter(graph.out_neighbors(node));
            preds.remove(&node);
            succ.remove(&node);
            graph.remove_edges_at_node(node);

            let mut twins = graph
                .out_neighbors(pred)
                .filter(|&v| {
                    v != node
                        && graph.in_neighbors(v).filter(|x| preds.contains(x)).count()
                            + (preds.contains(&v) as usize)
                            == preds.len()
                        && (graph.out_neighbors(v).filter(|x| succ.contains(x)).count()
                            + (succ.contains(&v) as usize))
                            == succ.len()
                })
                .collect_vec();

            graph.remove_edges_of_nodes(&twins);

            if !self.directed_mode {
                let mirrors = Self::get_mirrors(graph, node);

                if !mirrors.is_empty() {
                    twins.extend(&mirrors);

                    let mut found_dominated_nodes = false;
                    for u in mirrors {
                        let doms = self.remove_node_and_dominating(graph, u);
                        twins.extend(&doms);
                        found_dominated_nodes |= !doms.is_empty();
                    }

                    if found_dominated_nodes {
                        twins.sort_unstable();
                        twins.dedup();
                    }
                }
            }

            twins
        } else {
            graph.remove_edges_at_node(node);
            vec![]
        }
    }

    /// Computes mirror nodes as described proposed by Fomin et al. in "A measure & conquer approach
    /// for the analysis of exact algorithms". If we decide to put a node u into the solution, the
    /// set of mirrors needs to follow.
    fn get_mirrors(graph: &mut G, u: Node) -> Vec<Node> {
        let mut two_neighbors = BitSet::new_all_unset_but(
            graph.len(),
            graph
                .undir_neighbors(u)
                .flat_map(|v| graph.undir_neighbors(v))
                .map(|v| v as usize),
        );

        let nu =
            BitSet::new_all_unset_but(graph.len(), graph.undir_neighbors(u).map(|v| v as usize));
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

    /// Applies a series of kernelization rules and updates the `self.graph`, `self.partial_solution`
    /// as well as lower and upper bounds. If we solve or reject the instance during this process we
    /// return `Some( {Result} )`. If the instance remains unsolved, we return `None`.
    fn apply_kernelization(&mut self) -> Option<BBResult<G>> {
        if self.graph_is_reduced {
            return None;
        }

        let len_of_part_sol_before = self.partial_solution.len() as Node;
        loop {
            apply_rules_exhaustively(&mut self.graph, &mut self.partial_solution, false);
            if self.upper_bound + len_of_part_sol_before <= self.partial_solution.len() as Node {
                return Some(self.fail());
            }

            let pp_before = self.postprocessor.will_add();

            if apply_rule_undir_degree_two(&mut self.graph, &mut self.postprocessor)
                || apply_rule_funnel(
                    &mut self.graph,
                    &mut self.partial_solution,
                    &mut self.postprocessor,
                )
                || apply_rule_twins_degree_three(
                    &mut self.graph,
                    &mut self.partial_solution,
                    &mut self.postprocessor,
                )
                || apply_rule_desk(&mut self.graph, &mut self.postprocessor)
            {
                let pp_added = self.postprocessor.will_add() - pp_before;

                if pp_added > 0 {
                    if self.upper_bound <= pp_added {
                        return Some(self.fail());
                    }

                    self.upper_bound -= pp_added;
                    self.lower_bound = self.lower_bound.saturating_sub(pp_added);
                }

                continue;
            }

            if self.graph.number_of_nodes() > 2000 || thread_rng().gen_ratio(15, 16) {
                break;
            }

            let rule6_limit =
                self.upper_bound + len_of_part_sol_before - self.partial_solution.len() as Node + 1;

            match apply_rules_5_and_6(&mut self.graph, rule6_limit, &mut self.partial_solution) {
                None => return Some(self.fail()),
                Some(false) => break,   // no change
                Some(true) => continue, // did changes; restart
            };
        }
        let num_nodes_added_to_solution =
            self.partial_solution.len() as Node - len_of_part_sol_before;

        if num_nodes_added_to_solution >= self.upper_bound {
            return Some(self.fail());
        }

        if self.graph.number_of_edges() == 0 || self.graph.is_acyclic() {
            return Some(self.return_partial_solution_as_result());
        }

        self.upper_bound -= num_nodes_added_to_solution;
        self.lower_bound = self
            .lower_bound
            .saturating_sub(num_nodes_added_to_solution)
            .max(2);

        self.graph_is_connected = None;
        self.graph_is_reduced = true;

        None
    }

    /// Tries to split the instance into several SCCs (or remove acyclic subgraphs). In case of
    /// success, the functions setup the resume configuration and returns Some( {branching token} ).
    /// In case the computation should resume in the current branch, `None` is returned.
    fn try_split_into_sccs(&mut self) -> Option<BBResult<G>> {
        if self.graph_is_connected.unwrap_or(false) {
            return None;
        }

        let partition = self.graph.partition_into_strongly_connected_components();
        if partition.number_of_classes() == 1
            && partition.number_in_class(0) < self.graph.number_of_nodes()
        {
            let mut res = partition.split_into_subgraphs(&self.graph);
            let (subgraph, mapper) = res.pop().unwrap();
            assert!(res.is_empty());
            assert_eq!(self.graph.number_of_edges(), subgraph.number_of_edges());

            self.resume_with = ResumeMapped(mapper);

            let mut branch = self.branch(subgraph, self.lower_bound, self.upper_bound);
            branch.graph_is_reduced = true;
            branch.graph_is_connected = Some(true);
            return Some(BBResult::Branch(branch));
        }

        if partition.number_of_classes() > 1 {
            let mut subgraphs = partition.split_into_subgraphs(&self.graph);
            // shortest to the back since we want them process first and will pop frame the back
            subgraphs.sort_by_key(|(g, _)| self.graph.len() - g.len());

            let lower_bounds = subgraphs
                .iter()
                .map(|(graph, _): &(G, _)| -> Node {
                    let mut lb = LowerBound::new(graph);
                    if let Some(mc) = self.max_clique {
                        lb.set_max_clique(mc);
                    }
                    lb.compute();
                    lb.lower_bound().max(MIN_REDUCED_SCC_SIZE) as Node
                })
                .collect_vec();

            return Some(self.resume_scc_split(subgraphs, lower_bounds, None));
        }

        self.graph_is_connected = Some(true);

        None
    }

    fn try_branching_on_fancy_cuts(&mut self) -> Option<BBResult<G>> {
        if self.graph.len() < 64 {
            // We should never enter this branch for small graphs as we should have branched to the
            // matrix-solver. Nevertheless, fancy cut algorithms have weird corner cases and performance
            // issues for small graphs. Hence, we better skip them.
            return None;
        }

        let weak_articulation_point = if self.graph_is_connected.unwrap_or(false) {
            if let Some((ap, strong)) = compute_undirected_articulation_point(&self.graph) {
                if strong {
                    return Some(self.branch_on_node(ap));
                }

                Some(ap)
            } else {
                None
            }
        } else {
            None
        };

        if let Some((u, v)) = compute_undirected_articulation_pair(&self.graph, true) {
            return Some(self.resume_vertex_cut(
                None,
                vec![u, v],
                AllIntSubsets::new(2),
                true,
                None,
            ));
        }

        // search for cuts
        if let Some(cut) = compute_undirected_cut(&self.graph, self.upper_bound.min(5) - 1) {
            let k = cut.len();
            return Some(self.resume_vertex_cut(
                None,
                cut,
                AllIntSubsets::new(k as u32),
                true,
                None,
            ));
        }

        if let Some(mut cut) = self
            .graph
            .approx_min_balanced_cut(
                &mut rand::thread_rng(),
                30,
                0.25,
                Some(self.upper_bound.min(5)),
                true,
            )
            .or_else(|| {
                self.graph.approx_min_balanced_cut(
                    &mut rand::thread_rng(),
                    50,
                    10.0 / self.graph.number_of_nodes() as f64,
                    Some(self.upper_bound.min(3)),
                    true,
                )
            })
        {
            let k = cut.len();

            return Some(if k == 1 {
                // we got an articulation point, so let's directly branch on it
                self.branch_on_node(cut[0])
            } else {
                cut.sort_by_key(|&u| {
                    self.graph.len() * self.graph.len()
                        - self.graph.in_degree(u) as usize * self.graph.out_degree(u) as usize
                });
                self.resume_vertex_cut(None, cut, AllIntSubsets::new(k as u32), true, None)
            });
        }

        if let Some(ap) = weak_articulation_point {
            return Some(self.branch_on_node(ap));
        }

        None
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
