use super::*;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn branch_on_clique(&mut self, mut clique: Vec<Node>) -> BBResult<G> {
        if !BRANCH_ON_CLIQUES {
            unreachable!();
        }

        assert!(!clique.is_empty());

        if clique.len() == 1 {
            return self.branch_on_node(clique[0]);
        }

        if self.upper_bound < clique.len() as Node {
            return self.fail();
        }

        debug_assert!(clique.iter().all(|&u| clique
            .iter()
            .all(|&v| u == v || self.graph.has_undir_edge(u, v))));

        clique.sort_unstable_by_key(|&u| self.graph.total_degree(u) + self.graph.undir_degree(u));
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
                let mut remaining_nodes =
                    BitSet::new_all_set_but(self.graph.len(), from_child.iter().copied());

                for &u in &clique {
                    if self
                        .graph
                        .undir_neighbors(u)
                        .any(|v| remaining_nodes[v as usize])
                    {
                        continue;
                    }

                    remaining_nodes.set_bit(u as usize);
                    if self.graph.vertex_induced(&remaining_nodes).0.is_acyclic() {
                        remaining_nodes.not(); // now contains all node of solution!
                        return self.return_to_result_and_partial_solution(Some(
                            remaining_nodes.iter().map(|u| u as Node).collect_vec(),
                        ));
                    }
                    remaining_nodes.unset_bit(u as usize);
                }

                // So we did not find such a vertex: prepare further branching
                let solution_size = from_child.len() as Node;
                self.lower_bound = self.lower_bound.max(solution_size - 1);

                if self.upper_bound > solution_size {
                    // this does not have to hold, since we gave the branch some slack
                    self.upper_bound = solution_size;
                    current_best = Some(from_child);
                } else {
                    assert_eq!(self.upper_bound, solution_size);
                }
            } else {
                // If we cannot solve the delete-only-branch (i==0) within upper_bound+1, no other branch
                // can solve it in upper_bound! So we gave the first branch a +1 slack and can fail early
                // if the child branch cannot solve it
                if from_child.is_none() {
                    return self.fail();
                }
            }
        } else if let Some(from_child) = from_child {
            self.upper_bound = from_child.len() as Node;
            current_best = Some(from_child);
        }

        let mut subgraph = if i == clique.len() as Node {
            std::mem::take(&mut self.graph) // last iteration --- do not need the original graph anymore
        } else {
            self.graph.clone()
        };

        // delete and contract nodes as required by the branch
        let need_contraction = i > 0;

        let mut satellites = Vec::new();

        let nodes_deleted = if need_contraction {
            let mut nodes_deleted = clique.clone();
            let spare = nodes_deleted.swap_remove(i as usize - 1);
            subgraph.remove_edges_of_nodes(&nodes_deleted);

            if DELETE_TWINS_MIRRORS_AND_SATELLITES && self.get_mirrors(&subgraph, spare).is_empty()
            {
                satellites = self.get_satellite(&subgraph, spare);
            }

            let loops = subgraph.contract_node(spare); // will return at least all other nodes of clique
            subgraph.remove_edges_of_nodes(&loops);
            nodes_deleted.extend(&loops);

            for s in satellites {
                let loops = subgraph.contract_node(s);
                nodes_deleted.extend(&loops);
                subgraph.remove_edges_of_nodes(&loops);
            }

            nodes_deleted.sort_unstable();
            nodes_deleted.dedup();
            nodes_deleted
        } else {
            subgraph.remove_edges_of_nodes(&clique);
            clique.clone()
        };
        assert!(nodes_deleted.len() + 1 >= clique.len());

        // see above: give delete-only-branch more slack to fail early
        let slack = !need_contraction as Node;

        // can we already prune this branch?
        let num_nodes_deleted = nodes_deleted.len() as Node;
        if num_nodes_deleted >= self.upper_bound + slack {
            return self.resume_clique(current_best, clique, i + 1, None);
        }

        let mut branch = self.branch(
            subgraph,
            self.lower_bound.saturating_sub(num_nodes_deleted),
            self.upper_bound + slack - num_nodes_deleted,
        );
        branch.max_clique = self.max_clique;
        branch.partial_solution_parent = nodes_deleted;

        self.resume_with = ResumeClique(current_best, clique, i + 1);
        BBResult::Branch(branch)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;
    use rand::prelude::IteratorRandom;

    const SAMPLES_PER_GRAPH: usize = 6;

    fn stress_test<FB: Fn(AdjArrayUndir, Node) -> Frame<AdjArrayUndir>>(
        frame_builder: FB,
        expect_success: bool,
    ) {
        for_each_stress_graph(|filename, graph| {
            for i in 0..SAMPLES_PER_GRAPH {
                let clique_size = ((i % 3) + 2).min(graph.len());
                let clique = graph
                    .vertices()
                    .choose_multiple(&mut thread_rng(), clique_size);
                let mut graph = graph.clone();
                graph.connect_nodes(
                    &BitSet::new_all_unset_but(graph.len(), clique.iter().copied()),
                    false,
                );
                let opt_sol = branch_and_bound_matrix(&graph, None).unwrap().len() as Node;

                let mut frame = frame_builder(graph, opt_sol);
                let response = frame.branch_on_clique(clique.clone());
                let simulated_result = simulate_execution(&mut frame, response);

                if expect_success {
                    let my_size = simulated_result.as_ref().map_or(-1, |s| s.len() as isize);

                    assert_eq!(
                        my_size, opt_sol as isize,
                        "file: {} clique: {:?} opt: {} my-size: {} my-sol: {:?}",
                        filename, clique, opt_sol, my_size, simulated_result
                    );
                } else {
                    assert!(simulated_result.is_none())
                }
            }
        });
    }

    branching_stress_tests!(stress_test);

    #[test]
    #[ignore]
    fn interactive() {
        // data/stress_kernels/h_199_n19_m72_5a92308633d07ab45d5a0bbe8d54c7a3b7beee5dd581d4b5936fe15e7570f0d9_kernel.metis clique: [5, 15]
        let mut graph = AdjArrayUndir::try_read_graph(FileFormat::Metis, std::path::Path::new("data/stress_kernels/h_199_n10_m27_2f4c75936994bf6310e843c04f0c08f479a5f7339ec1e807b72f16ac65817f65_kernel.metis")).unwrap();
        let clique = vec![0, 8]; // 0,9

        graph.connect_nodes(
            &BitSet::new_all_unset_but(graph.len(), clique.iter().copied()),
            false,
        );

        println!("{:?}", &graph);

        let mut frame = Frame::new(graph.clone(), 2, 3);
        let response = frame.branch_on_clique(clique.clone());
        let simulated_result = simulate_execution(&mut frame, response);
        assert_eq!(simulated_result.unwrap().len(), 2);
    }
}
