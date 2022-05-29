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
            let mut applied = false;

            macro_rules! apply_sure_rule {
                ($_k:ident,  $r:expr) => {{
                    let edges = self.graph.number_of_edges();
                    if ($r) {
                        applied = true;
                        while (apply_rule_1(&mut self.graph, &mut self.partial_solution)
                            | apply_rule_4(&mut self.graph, &mut self.partial_solution))
                            || apply_rule_scc(&mut self.graph)
                        {}

                        true
                    } else {
                        assert_eq!(self.graph.number_of_edges(), edges);
                        false
                    }
                }};
            }

            macro_rules! apply_rule {
                ($k:ident, $r:expr) => {
                    paste::item! {{
                        let misses = &mut self.kernel_misses.[< $k >];

                        if !USE_ADAPTIVE_KERNEL_FREQUENCY || *misses == 0 || rand::thread_rng().gen_ratio(1, *misses + 1) {
                            if (apply_sure_rule!($k, $r)) {
                                *misses = 0;
                            } else {
                                *misses += 1;
                            }
                        }
                    }}
                };
            }

            while (apply_rule_1(&mut self.graph, &mut self.partial_solution)
                | apply_rule_4(&mut self.graph, &mut self.partial_solution))
                || apply_rule_scc(&mut self.graph)
            {}

            if TRIVIAL_KERNEL_RULES_ONLY {
                break;
            }

            if self.upper_bound + len_of_part_sol_before <= self.partial_solution.len() as Node {
                return Some(self.fail());
            }

            if !self.directed_mode {
                apply_sure_rule!(
                    di_cliques,
                    apply_rule_di_cliques(&mut self.graph, &mut self.partial_solution)
                );
                apply_sure_rule!(pie, apply_rule_pie(&mut self.graph));
                apply_rule!(
                    c4,
                    apply_rule_c4(&mut self.graph, &mut self.partial_solution)
                );
                apply_rule!(
                    undir_dom_node,
                    apply_rule_dominance(&mut self.graph, &mut self.partial_solution)
                );
            }
            apply_rule!(
                domn,
                apply_rule_domn(&mut self.graph, &mut self.partial_solution)
            );
            apply_rule!(redundant_cycle, apply_rule_redundant_cycle(&mut self.graph));
            if !self.directed_mode {
                apply_rule!(
                    unconfined,
                    apply_rule_unconfined(&mut self.graph, &mut self.partial_solution)
                );
                apply_rule!(
                    crown,
                    apply_rule_crown(&mut self.graph, &mut self.partial_solution, None)
                );
            }
            apply_rule!(dom_node_edges, apply_rule_dome(&mut self.graph));

            if self.upper_bound + len_of_part_sol_before <= self.partial_solution.len() as Node {
                return Some(self.fail());
            }

            if !self.directed_mode {
                let pp_before = self.postprocessor.will_add();

                apply_sure_rule!(
                    undir_degree_two,
                    apply_rule_undir_degree_two(&mut self.graph, &mut self.postprocessor)
                );
                apply_sure_rule!(
                    funnel,
                    apply_rule_funnel(
                        &mut self.graph,
                        &mut self.partial_solution,
                        &mut self.postprocessor,
                    )
                );
                apply_sure_rule!(
                    twins_degree_three,
                    apply_rule_twins_degree_three(
                        &mut self.graph,
                        &mut self.partial_solution,
                        &mut self.postprocessor,
                    )
                );
                apply_sure_rule!(
                    desk,
                    apply_rule_desk(&mut self.graph, &mut self.postprocessor)
                );

                let pp_added = self.postprocessor.will_add() - pp_before;
                if pp_added >= self.upper_bound {
                    return Some(self.fail());
                }
                self.upper_bound -= pp_added;
                self.lower_bound = self.lower_bound.saturating_sub(pp_added);
            }

            let rule6_limit =
                self.upper_bound + len_of_part_sol_before - self.partial_solution.len() as Node;

            apply_rule!(rules56, {
                let result =
                    apply_rules_5_and_6(&mut self.graph, rule6_limit, &mut self.partial_solution);
                if result.is_none() {
                    return Some(self.fail());
                }
                result.unwrap()
            });

            if !applied {
                break;
            }
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

        self.directed_mode = self
            .graph
            .vertices()
            .all(|u| self.graph.undir_degree(u) == 0);

        None
    }
}
