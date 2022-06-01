use super::*;

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

            let ub = subgraph.number_of_nodes();
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

            if scc.number_of_nodes() < MIN_NODES_FOR_CACHE && MIN_NODES_FOR_CACHE < 128 {
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

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    const SAMPLES_PER_GRAPH: Node = 5;

    fn stress_test<FB: Fn(AdjArrayUndir, Node) -> Frame<AdjArrayUndir>>(
        frame_builder: FB,
        expect_success: bool,
    ) {
        for_each_stress_graph(|filename, graph| {
            'outer: for i in 0..SAMPLES_PER_GRAPH {
                let mut graph = graph.clone();
                let num_sccs = (1 + 2 * (i % 3)).min(graph.number_of_nodes() / 5);
                if num_sccs == 0 {
                    break;
                }

                let mut nodes = graph
                    .vertices()
                    .filter(|&u| graph.total_degree(u) > 0)
                    .collect_vec();
                let mut nodes_deleted = graph
                    .vertices()
                    .filter(|&u| graph.total_degree(u) == 0)
                    .collect_vec();
                loop {
                    if nodes.is_empty() {
                        continue 'outer;
                    }

                    let mut sccs = graph
                        .partition_into_strongly_connected_components()
                        .split_into_subgraphs(&graph);
                    let mut num_accepted = 0;
                    while let Some((scc, mapper)) = sccs.pop() {
                        if scc.number_of_nodes() < 3
                            || branch_and_bound_matrix(&scc, None).unwrap().len() < 2
                        {
                            // within the B&B algorithm we maintain the invariant that each SCC has a lower bound of at least 2
                            for x in scc.vertices().map(|u| mapper.old_id_of(u).unwrap()) {
                                nodes_deleted.push(x);
                                graph.remove_edges_at_node(x);
                            }
                        } else {
                            num_accepted += 1;
                        }
                    }

                    if num_accepted >= num_sccs {
                        break;
                    }

                    let node = nodes.swap_remove(thread_rng().gen_range(0..nodes.len()));
                    nodes_deleted.push(node);
                    graph.remove_edges_at_node(node);
                }

                let opt_sol = branch_and_bound_matrix(&graph, None).unwrap().len() as Node;

                let mut frame = frame_builder(graph.clone(), opt_sol);
                let response = frame.try_branch_on_sccs();
                if response.is_none() {
                    assert_eq!(num_sccs, 1);
                    assert!(nodes_deleted.is_empty());
                    continue 'outer;
                }
                let response = response.unwrap();
                let simulated_result = simulate_execution(&mut frame, response);

                if expect_success {
                    let my_size = simulated_result.as_ref().map_or(-1, |s| s.len() as isize);

                    assert_eq!(
                        my_size, opt_sol as isize,
                        "file: {} nodes_deleted: {:?} opt: {} my-size: {} my-sol: {:?} num_sccs: {}",
                        filename, nodes_deleted, opt_sol, my_size, simulated_result, graph.partition_into_strongly_connected_components().number_of_classes()
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
        let mut graph = AdjArrayUndir::try_read_graph(FileFormat::Metis, std::path::Path::new("data/stress_kernels/h_199_n12_m82_018d0156f945cd3eb240dd2eece4dab1a5786009e1f0f242fbee75b3f83be416_kernel.metis")).unwrap();
        let nodes_deleted = vec![0, 1, 3];
        let opt = 8;

        graph.remove_edges_of_nodes(&nodes_deleted);
        println!("{:?}", &graph);
        assert_eq!(
            branch_and_bound_matrix(&graph, None).unwrap().len() as Node,
            opt
        );
        let mut frame = Frame::new(graph, 0, opt + 1);
        let response = frame.try_branch_on_sccs().unwrap();
        let simulated_result = simulate_execution(&mut frame, response);
        assert_eq!(simulated_result.unwrap().len() as Node, opt);
    }
}
