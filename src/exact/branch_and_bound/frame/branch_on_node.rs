use super::*;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn branch_on_node(&mut self, node: Node) -> BBResult<G> {
        assert!(self.graph.in_degree(node) > 0 && self.graph.out_degree(node) > 0);

        let mut graph = self.graph.clone();

        // remove vertex and its mirrors
        let mut partial_solution = vec![node];
        if DELETE_TWINS_MIRRORS_AND_SATELLITES {
            partial_solution.extend(&self.get_mirrors(&graph, node));
        }
        graph.remove_edges_of_nodes(&partial_solution);

        // check that our solution can met the upper_bound
        let num_nodes_deleted = partial_solution.len() as Node;
        if num_nodes_deleted >= self.upper_bound {
            return self.resume_delete_branch(node, num_nodes_deleted, None);
        }

        // branch!
        self.resume_with = ResumeDeleteBranch(node, num_nodes_deleted);
        let mut branch = self.branch(
            graph,
            self.lower_bound.saturating_sub(num_nodes_deleted),
            self.upper_bound - num_nodes_deleted,
        );
        branch.partial_solution_parent = partial_solution;

        BBResult::Branch(branch)
    }

    pub(super) fn resume_delete_branch(
        &mut self,
        node: Node,
        num_nodes_deleted: Node,
        mut from_child: Option<OptSolution>,
    ) -> BBResult<G> {
        if let Some(from_child) = from_child.as_mut() {
            if let Some(sol) = from_child.as_mut() {
                debug_assert!(self.upper_bound > sol.len() as Node);
                self.upper_bound = sol.len() as Node;
                self.lower_bound = self.lower_bound.max(sol.len() as Node - num_nodes_deleted);

                if self.upper_bound <= self.lower_bound {
                    return self.return_to_result_and_partial_solution(Some(sol.clone()));
                }
            } else {
                self.lower_bound = self.upper_bound.saturating_sub(num_nodes_deleted);
            }
        }

        // if contraction is no option, we can skip the branch
        if self.graph.has_edge(node, node) || self.graph.undir_degree(node) >= self.upper_bound {
            return self.return_to_result_and_partial_solution(from_child.unwrap_or(None));
        }

        // Setup child call
        let mut subgraph = std::mem::take(&mut self.graph);

        let satellites = if DELETE_TWINS_MIRRORS_AND_SATELLITES && num_nodes_deleted == 1 {
            Some(self.get_satellite(&subgraph, node).collect_vec())
        } else {
            None
        };

        let mut removed = subgraph.contract_node(node);
        subgraph.remove_edges_of_nodes(&removed);

        if let Some(satellites) = satellites {
            for s in satellites {
                let loops = subgraph.contract_node(s);
                removed.extend(&loops);
                subgraph.remove_edges_of_nodes(&loops);
            }
            removed.sort_unstable();
            removed.dedup();
        }

        if self.upper_bound <= removed.len() as Node {
            return self.return_to_result_and_partial_solution(from_child.unwrap_or(None));
        }

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

        // Setup re-entry
        self.resume_with = ResumeContractBranch(from_child.unwrap_or(None));

        BBResult::Branch(branch)
    }

    pub(super) fn resume_contract_branch(
        &mut self,
        from_delete: OptSolution,
        from_child: OptSolution,
    ) -> BBResult<G> {
        self.return_to_result_and_partial_solution(from_child.or(from_delete))
    }
}
