use super::*;
use crate::heuristics::weakest_link::weakest_link;

impl<G: BnBGraph> Frame<G> {
    /// Tries to split the instance into several SCCs (or remove acyclic subgraphs). In case of
    /// success, the functions setup the resume configuration and returns Some( {branching token} ).
    /// In case the computation should resume in the current branch, `None` is returned.
    pub(super) fn try_branch_on_sccs(&mut self) -> Option<BBResult<G>> {
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

            let ub = subgraph.number_of_nodes().saturating_sub(1);
            assert!(ub >= self.lower_bound);
            let mut branch = self.branch(subgraph, self.lower_bound, self.upper_bound.min(ub));
            branch.graph_is_reduced = true;
            branch.graph_is_connected = Some(true);
            return Some(BBResult::Branch(branch));
        }

        if partition.number_of_classes() > 1 {
            let mut subgraphs = partition.split_into_subgraphs(&self.graph);
            // shortest to the back since we want them process first and will pop frame from the back
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

    pub(super) fn resume_scc_split(
        &mut self,
        mut subgraphs: Vec<(G, NodeMapper)>,
        mut lower_bounds: Vec<Node>,
        from_child: Option<OptSolution>,
    ) -> BBResult<G> {
        if let Some(opt_sol) = from_child {
            if let Some(sol) = opt_sol {
                self.upper_bound -= sol.len() as Node;
                self.lower_bound = self.lower_bound.saturating_sub(sol.len() as Node);

                let mapper = subgraphs.pop().unwrap().1;
                self.partial_solution
                    .extend(sol.into_iter().map(|u| mapper.old_id_of(u).unwrap()));

                assert_eq!(subgraphs.len(), lower_bounds.len());
            } else {
                return self.fail();
            }
        }

        let frame_lb = if subgraphs.len() == 1 {
            self.lower_bound
        } else {
            0
        };

        if let Some(next_branch) = subgraphs.last_mut() {
            let lower_bound = lower_bounds.pop().unwrap().max(frame_lb);
            let remaining_lower = lower_bounds.iter().sum::<Node>();

            if remaining_lower + lower_bound >= self.upper_bound {
                return self.fail();
            }

            let scc = std::mem::take(&mut next_branch.0);

            let upper_bound = (self.upper_bound - remaining_lower).min(scc.number_of_nodes());

            if scc.number_of_nodes() < MIN_NODES_FOR_CACHE {
                if let Some(sol) =
                    branch_and_bound_matrix_lower(&scc, lower_bound, Some(upper_bound - 1))
                {
                    return self.resume_scc_split(subgraphs, lower_bounds, Some(Some(sol)));
                } else {
                    return self.fail();
                }
            }

            let mut branch = self.branch(scc, lower_bound, upper_bound);
            branch.graph_is_connected = Some(true);

            self.resume_with = ResumeSCCSplit(subgraphs, lower_bounds);

            return BBResult::Branch(branch);
        }

        self.return_partial_solution_as_result()
    }

    pub(super) fn resume_mapped(
        &mut self,
        mapper: NodeMapper,
        mut from_child: OptSolution,
    ) -> BBResult<G> {
        if let Some(sol) = from_child.as_mut() {
            for u in sol {
                *u = mapper.old_id_of(*u).unwrap();
            }
        }

        self.return_to_result_and_partial_solution(from_child)
    }
}
