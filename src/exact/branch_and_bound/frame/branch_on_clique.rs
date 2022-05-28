use super::*;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn branch_on_clique(&mut self, mut clique: Vec<Node>) -> BBResult<G> {
        if !BRANCH_ON_CLIQUES {
            unreachable!();
        }

        clique.sort_unstable_by_key(|&u| -(self.graph.total_degree(u) as i64));
        self.resume_clique(None, clique, 0, None)
    }

    pub(super) fn resume_clique(
        &mut self,
        mut current_best: OptSolution,
        clique: Vec<Node>,
        i: Node,
        from_child: OptSolution,
    ) -> BBResult<G> {
        // Last branch: return best solution
        if i > clique.len() as Node {
            return self.return_to_result_and_partial_solution(from_child.or(current_best));
        }

        // if i==1 we just received the result from the first (delete only) branch; this requires
        // some special care to fail early or to prune more efficiently
        if i == 1 {
            if let Some(from_child) = from_child {
                // If we find a single vertex of the clique, that we can reinsert without introducing a
                // new cycle, we do not have to branch on any more cases
                let mut solution =
                    BitSet::new_all_unset_but(self.graph.len(), from_child.iter().copied());

                for &u in &clique {
                    if self.graph.undir_degree(u) >= clique.len() as Node && // skip if all undir neighbors are inside clique
                        self.graph.undir_neighbors(u).any(|v| !solution[v as usize])
                    {
                        continue;
                    }

                    solution.unset_bit(u as usize);
                    if self.graph.vertex_induced(&solution).0.is_acyclic() {
                        return self.return_to_result_and_partial_solution(Some(
                            solution.iter().map(|u| u as Node).collect_vec(),
                        ));
                    }
                    solution.set_bit(u as usize);
                }

                // So we did not find such a vertex: prepare further branching
                let solution_size = from_child.len() as Node;
                self.lower_bound = self.lower_bound.max(solution_size - 1);

                if self.upper_bound > solution_size {
                    // this does not have to hold, since we gave the branch some slack
                    self.upper_bound = solution_size;
                    current_best = Some(from_child);
                } else {
                    debug_assert_eq!(self.upper_bound + 1, solution_size);
                }
            } else {
                // If we cannot solve the delete-only-branch (i==0) within upper_bound+1, no other branch
                // can solve it in upper_bound! So we gave the first branch a +1 slack and can fail early
                // if the child branch cannot solve it
                if from_child.is_none() {
                    return self.fail();
                }
            }
        } else if from_child.is_some() {
            // a contraction branch can only ever be better by 1 than the delete-only branch; so
            // if we are here, we are done
            assert_eq!(self.lower_bound, from_child.as_ref().unwrap().len() as Node);

            return self.return_to_result_and_partial_solution(from_child);
        }

        let mut subgraph = if i == clique.len() as Node {
            std::mem::take(&mut self.graph) // last iteration --- do not need the original graph anymore
        } else {
            self.graph.clone()
        };

        // delete and contract nodes as required by the branch
        let need_contraction = i > 0;

        let nodes_deleted = if need_contraction {
            let spare = clique[i as usize - 1];
            subgraph.contract_node(spare) // will return at least all other nodes of clique
        } else {
            clique.clone()
        };
        assert!(nodes_deleted.len() + 1 >= clique.len());

        subgraph.remove_edges_of_nodes(&nodes_deleted);

        // can we already prune this branch?
        let num_nodes_deleted = nodes_deleted.len() as Node;
        if num_nodes_deleted >= self.upper_bound {
            return self.resume_clique(current_best, clique, i + 1, None);
        }

        // see above: give delete-only-branch more slack to fail early
        let slack = !need_contraction as Node;

        let mut branch = self.branch(
            subgraph,
            self.lower_bound.saturating_sub(num_nodes_deleted),
            self.upper_bound + slack - num_nodes_deleted,
        );
        assert_eq!(self.max_clique, Some(clique.len() as Node));
        branch.max_clique = self.max_clique;
        branch.partial_solution_parent = nodes_deleted;

        self.resume_with = ResumeClique(current_best, clique, i + 1);
        BBResult::Branch(branch)
    }
}
