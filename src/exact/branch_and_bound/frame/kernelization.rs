use super::*;

impl<G: BnBGraph> Frame<G> {
    /// Applies a series of kernelization rules and updates the `self.graph`, `self.partial_solution`
    /// as well as lower and upper bounds. If we solve or reject the instance during this process we
    /// return `Some( {Result} )`. If the instance remains unsolved, we return `None`.
    pub(super) fn apply_kernelization(&mut self) -> Option<BBResult<G>> {
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
}
