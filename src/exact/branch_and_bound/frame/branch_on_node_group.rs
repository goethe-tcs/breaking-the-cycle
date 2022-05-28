use super::*;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn branch_on_node_group(&mut self, group: Vec<Node>) -> BBResult<G> {
        if !SEARCH_CUTS && !BRANCH_ON_CYCLES {
            unreachable!();
        }

        assert!(!group.is_empty());
        assert!(group
            .iter()
            .all(|&u| self.graph.in_degree(u) > 0 && self.graph.out_degree(u) > 0));

        if group.len() == 1 {
            self.branch_on_node(*group.first().unwrap())
        } else {
            let mut branches = group
                .iter()
                .copied()
                .powerset()
                .filter(|b| !b.is_empty())
                .collect_vec();

            branches.iter_mut().for_each(|b| b.sort_unstable());
            branches.sort_unstable_by_key(|b| b.len());

            self.resume_node_group(None, branches, group, None, None)
        }
    }

    pub(super) fn resume_node_group(
        &mut self,
        mut current_best: OptSolution,
        mut branches_left: Vec<Vec<Node>>,
        group: Vec<Node>,
        mut deleted_for_child: Option<Vec<Node>>,
        solution_from_child: OptSolution,
    ) -> BBResult<G> {
        if let Some(mut sol) = solution_from_child {
            let deleted_for_child = deleted_for_child.as_mut().unwrap();

            if deleted_for_child == &group {
                // result is from contraction-free branch, so we can derive a lower bound from it
                // note that the reinsertable-optimization down below wont improve the lower bound
                // so we can update it here
                let lb = sol.len() - deleted_for_child.len();
                self.lower_bound = self.lower_bound.max(lb as Node);
            }

            if self.upper_bound as usize <= sol.len() {
                assert_eq!(deleted_for_child.len(), group.len());
                return self.fail();
            }

            // we try to improve the solution computed; if --by chance-- the optimal solution obtained
            // is compatible with the reinsertion of nodes deleted, we do so. this also prunes the
            // branches for this smaller group
            let reinsertable = deleted_for_child
                .iter()
                .copied()
                .filter(|&u| {
                    !self
                        .graph
                        .is_node_on_cycle_after_deleting(u, sol.iter().copied().filter(|&v| v != u))
                })
                .collect_vec();

            // observe that `reinsertable` contains those nodes that can be reinserted in isolation;
            // we may even be able to combine them --- that's what we're checking next:
            if !reinsertable.is_empty() {
                let largest_reinsertable_group = if reinsertable.len() == 1 {
                    // compute smaller cut without the reinsertable node ...
                    let u = reinsertable[0];
                    let mut smaller_group = deleted_for_child.clone();
                    smaller_group.remove(deleted_for_child.iter().position(|&x| x == u).unwrap());

                    // ... and remove it from the future branches
                    let pos = branches_left.iter().position(|x| x == &smaller_group);
                    if let Some(pos) = pos {
                        branches_left.swap_remove(pos);
                    }

                    reinsertable
                } else {
                    let solution = BitSet::new_all_set_but(self.graph.len(), sol.iter().copied());

                    // we compute the largest subset of reinsertables that leads to an acyclic solution;
                    // in the process, we will delete all acyclic subsets from the future branches
                    let mut max_grp: Option<Vec<Node>> = None;
                    for grp in reinsertable.iter().copied().powerset() {
                        let mut solution = solution.clone();
                        for &x in &grp {
                            solution.set_bit(x as usize);
                        }
                        if !self.graph.vertex_induced(&solution).0.is_acyclic() {
                            continue;
                        }

                        let smaller_group = deleted_for_child
                            .iter()
                            .copied()
                            .filter(|u| !grp.contains(u))
                            .collect_vec();

                        debug_assert!(smaller_group.iter().is_sorted());
                        let pos = branches_left.iter().position(|x| x == &smaller_group);
                        if let Some(pos) = pos {
                            branches_left.swap_remove(pos);
                        }

                        if max_grp.as_ref().map_or(true, |g| g.len() < grp.len()) {
                            max_grp = Some(grp.clone());
                        }
                    }
                    max_grp.unwrap()
                };

                // given the largest subset, improve the solution
                for u in largest_reinsertable_group {
                    let pos = sol.iter().position(|&x| x == u).unwrap();
                    sol.swap_remove(pos);

                    let pos = deleted_for_child.iter().position(|&x| x == u).unwrap();
                    deleted_for_child.swap_remove(pos);
                }
            }

            debug_assert!(sol.len() < self.upper_bound as usize);
            self.upper_bound = sol.len() as Node;

            current_best = Some(sol);
            if self.upper_bound == self.lower_bound {
                return self.return_to_result_and_partial_solution(current_best);
            }

            if (current_best.as_ref().unwrap().len() - deleted_for_child.len()) as Node
                == self.lower_bound
            {
                // we cannot improve the solution with any cuts of the same size (or larger)
                // than the cut we just got
                branches_left = branches_left
                    .into_iter()
                    .filter(|b| b.len() < deleted_for_child.len())
                    .collect_vec();
            }
        } else if deleted_for_child.as_ref().map_or(false, |g| g == &group) {
            // if we did not receive a solution from the delete-only branch, we still know that
            // it's size is at least self.upper_bound; thus we can at least derive a lower bound
            // from it
            self.lower_bound = self.lower_bound.max(self.upper_bound - group.len() as Node);
        }

        while let Some(nodes_to_delete) = branches_left.pop() {
            let mut graph = self.graph.clone();
            let mut partial_solution = nodes_to_delete.clone();

            let nodes_to_spare = group
                .iter()
                .copied()
                .filter(|u| !nodes_to_delete.contains(u))
                .collect_vec();

            for &u in &nodes_to_spare {
                let loops = graph.contract_node(u);
                graph.remove_edges_of_nodes(&loops);
                partial_solution.extend(&loops);
            }

            graph.remove_edges_of_nodes(&nodes_to_delete);

            partial_solution.sort_unstable();
            partial_solution.dedup();

            if self.upper_bound <= partial_solution.len() as Node {
                continue;
            }

            let mut branch = self.branch(
                graph,
                self.lower_bound
                    .saturating_sub(partial_solution.len() as Node),
                self.upper_bound - partial_solution.len() as Node,
            );
            branch.partial_solution_parent = partial_solution;

            self.resume_with = ResumeNodeGroup(current_best, branches_left, group, nodes_to_delete);

            return BBResult::Branch(branch);
        }

        self.return_to_result_and_partial_solution(current_best)
    }
}
