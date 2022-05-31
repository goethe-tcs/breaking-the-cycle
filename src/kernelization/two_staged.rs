use super::*;
use crate::bitset::BitSet;
use arrayvec::ArrayVec;
use itertools::Itertools;

/// This rule is inspired by an observation of [Chen et al., "Vertex Cover: Further
/// Observations and Further Improvements"] which they call "vertex folding". Translated to DFVS
/// the idea is roughly:
/// Let `u`, `v`, `w` be an undirected path with the following properties:
///   - node `v` has in-deg = out-deg = undir-deg = 2
///   - there are no edges between `u` and `w`
///   - at most one of `u` and `w` may have directed edges
///
/// Consider the path `u` <=> `v` <=> `w`; up to symmetry there are three possible DFVS
/// `{v}`, `{u, w}`, and `{v, w}` (analogously to `{u, v}`).
/// No observe that `{u, w}` is strictly better than `{v, w}` in the sense that both cover the
/// center node `v` but `{u, w}` potentially covers additional cycles at `u`. In other words: any
/// DFVS that contains `{v, w}` is still legal if we replace `v` by `u` --- the opposite is, however,
/// not true.
///
/// Thus we only need to consider two cases namely `{v}` and `{u, w}`, i.e. we need one or two nodes
/// of the two-path in our solution. However, we can only decide which of these cases applies after
/// the global optimization problem has been solved. We account for this as follows:
///
/// We fold the two-path into node `u`, this is done by removing `v` and copying over all edges
/// from `w` to `u` and removing `w`. For clarity denote the new node as `u'` and the original one
/// as `u` (in the implementation they have the same id!). If `u'` is not part of the solution
/// for the kernel, then we know that all cycles at `u` and `w` have been removed without touching
/// the nodes themselves. So it suffices to add `{v}` to the solution. If, however, `u'` is
/// part of the solution, then we replace it with `{u, w}`.
pub fn apply_rule_undir_degree_two<G>(graph: &mut G, postprocessor: &mut Postprocessor) -> bool
where
    G: ReducibleGraph,
{
    repeat_while(|| {
        let mut applied = false;
        for v in graph.vertices_range() {
            if graph.undir_degree(v) != 2 || graph.has_dir_edges(v) {
                continue;
            }

            let (u, w) = {
                let mut neigh = graph.out_neighbors(v);
                let u = neigh.next().unwrap();
                let w = neigh.next().unwrap();
                if graph.total_degree(u) > graph.total_degree(w) {
                    (u, w)
                } else {
                    (w, u)
                }
            };

            if graph.has_dir_edges(u) && graph.has_dir_edges(w)
                || graph.has_edge(u, w)
                || graph.has_edge(w, u)
            {
                continue;
            }

            applied = true;
            postprocessor.push(Box::new(PostRuleUndirDegreeTwo {
                search: u,
                if_found: w,
                if_not_found: v,
            }));

            // remove nodes v and w and copy over the edges from w to u
            graph.remove_edges_at_node(v);

            let out_n = graph.out_neighbors(w).collect_vec();
            let in_n = graph.in_neighbors(w).collect_vec();

            graph.remove_edges_at_node(w);

            for x in in_n {
                graph.try_add_edge(x, u);
            }

            for x in out_n {
                graph.try_add_edge(u, x);
            }
        }

        applied
    })
}

#[derive(Debug)]
struct PostRuleUndirDegreeTwo {
    search: Node,
    if_found: Node,
    if_not_found: Node,
}

impl PostprocessorRule for PostRuleUndirDegreeTwo {
    fn apply(&mut self, fvs: &mut Vec<Node>) {
        fvs.push(if fvs.contains(&self.search) {
            self.if_found
        } else {
            self.if_not_found
        })
    }

    fn will_add(&self) -> Node {
        1
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This rule is due to Xiao and Nagamochi "Confining sets and avoiding bottleneck cases: A simple
/// maximum independent set algorithm in degree-3 graphs".
///
/// We implement an adaptation of Hespe et al. "WeGotYouCovered: The Winning Solver from the PACE 2019
///  Implementation Challenge, Vertex Cover Track" (section 4.1 / "Twins")
///
/// Let `u` and `v` be vertices of degree 3 with `N(u) = N(v)`. If `G[N(u)]` has |a cycle|, then add
/// `N(u)` to the minimum vertex cover and remove `u`, `v`, `N(u)`, `N(v)` from G. Otherwise, some
/// `u` and `v` may belong to some minimum vertex cover. We still remove `u`, `v`, `N(u)` and `N(v)`
/// from G, and add a new gadget vertex `w` to `G` with edges to `u`’s two-neighborhood (vertices at
/// a distance 2 from `u`).  If `w` is in the computed minimum vertex cover, then `u`’s (and `v`’s)
/// neighbors are in some minimum vertex cover, otherwise u and v are in a minimum vertex cover.
pub fn apply_rule_twins_degree_three<G>(
    graph: &mut G,
    dfvs: &mut Vec<Node>,
    postprocessor: &mut Postprocessor,
) -> bool
where
    G: ReducibleGraph,
{
    repeat_while(|| {
        let mut candidates = graph
            .vertices()
            .filter_map(|u| {
                if graph.undir_degree(u) == 3 && !graph.has_dir_edges(u) {
                    let mut ns: ArrayVec<Node, 3> = graph.out_neighbors(u).collect();
                    ns.sort_unstable();

                    return Some((u, (ns[0], ns[1], ns[2])));
                }

                None
            })
            .collect_vec();

        if candidates.is_empty() {
            return false;
        }

        candidates.sort_by_key(|(_, ns)| *ns);

        for (&(u, nu), &(v, nv)) in candidates.iter().tuple_windows() {
            if nu != nv || graph.has_edge(u, v) {
                continue;
            }

            let (w1, w2, w3): (Node, Node, Node) = nu;

            let cycle_in_neighbors = graph.has_undir_edge(w1, w2)
                || graph.has_undir_edge(w2, w3)
                || graph.has_undir_edge(w3, w1)
                || graph.has_edge(w1, w2) && graph.has_edge(w2, w3) && graph.has_edge(w3, w1)
                || graph.has_edge(w1, w3) && graph.has_edge(w3, w2) && graph.has_edge(w2, w1);

            if cycle_in_neighbors {
                // twins can safely be deleted as we need to remove the neighbors
                dfvs.extend([w1, w2, w3].iter().copied());
                graph.remove_edges_of_nodes(&[u, v, w1, w2, w3]);
                return true;
            } else {
                let mut two_neighbors_out = BitSet::new_all_unset_but(
                    graph.len(),
                    graph
                        .out_neighbors(u)
                        .flat_map(|v| graph.out_neighbors(v).map(|w| w as usize)),
                );

                let mut two_neighbors_in = BitSet::new_all_unset_but(
                    graph.len(),
                    graph
                        .out_neighbors(u)
                        .flat_map(|v| graph.in_neighbors(v).map(|w| w as usize)),
                );

                two_neighbors_out.unset_bit(u as usize);
                two_neighbors_out.unset_bit(v as usize);
                two_neighbors_in.unset_bit(u as usize);
                two_neighbors_in.unset_bit(v as usize);

                graph.remove_edges_at_node(u);
                for v in two_neighbors_out.iter().map(|v| v as Node) {
                    graph.add_edge(u, v);
                }

                for v in two_neighbors_in.iter().map(|v| v as Node) {
                    graph.add_edge(v, u);
                }
                graph.remove_edges_of_nodes(&[v, w1, w2, w3]);

                postprocessor.push(Box::new(PostRuleTwinsDegreeThree {
                    search: u,
                    if_found: nu,
                    if_not_found: v,
                }));

                return true;
            }
        }

        false
    })
}

#[derive(Debug)]
struct PostRuleTwinsDegreeThree {
    search: Node,
    if_found: (Node, Node, Node),
    if_not_found: Node,
}

impl PostprocessorRule for PostRuleTwinsDegreeThree {
    fn apply(&mut self, fvs: &mut Vec<Node>) {
        if let Some(i) = fvs.iter().position(|&x| x == self.search) {
            fvs[i] = self.if_found.0;
            fvs.push(self.if_found.1);
            fvs.push(self.if_found.2);
        } else {
            fvs.push(self.search);
            fvs.push(self.if_not_found);
        }
    }

    fn will_add(&self) -> Node {
        2
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This rule is due to Hespe et al. "WeGotYouCovered: The Winning Solver from the PACE 2019
/// Implementation Challenge, Vertex Cover Track" (section 4.1 / "alternative")
///
/// We search for a cordless cycle (a1 b1 a2 b2) where each node has degree at least three.
/// Let A={a1, a2} and B={b1, b2}. We require
///  - |N(A) \ B| <= 2
///  - |N(B) \ A| <= 2
///  - intersection(N(A), N(B)) is empty
///
/// Then we remove the A, B and connect (N(A) \ B) x (N(B) \ A). If (N(A) \ B) is contained in
/// the global solution, we add nodes B to the solution, otherwise A.
pub fn apply_rule_desk<G: ReducibleGraph>(
    graph: &mut G,
    postprocessor: &mut Postprocessor,
) -> bool {
    repeat_while(|| {
        let c4 = (|| {
            let node_qualifies =
                |u| !graph.has_dir_edges(u) && (3..=4).contains(&graph.undir_degree(u));

            let compute_neighbor_union = |u, v| {
                let mut nb_union = graph.undir_neighbors(u).collect_vec();
                for x in graph.undir_neighbors(v) {
                    if !nb_union.contains(&x) {
                        nb_union.push(x)
                    }
                }
                nb_union
            };

            for a1 in graph.vertices() {
                if !node_qualifies(a1) {
                    continue;
                }

                for b1b2 in graph
                    .undir_neighbors(a1)
                    .filter(|&x| node_qualifies(x))
                    .combinations(2)
                {
                    let (b1, b2) = (b1b2[0], b1b2[1]);
                    if graph.has_edge(b1, b2) || graph.has_edge(b2, b1) {
                        continue;
                    }

                    let mut nb_union = compute_neighbor_union(b1, b2);
                    nb_union.swap_remove(nb_union.iter().position(|&x| x == a1).unwrap());
                    if nb_union.len() > 3 {
                        continue;
                    }

                    for a2 in graph.undir_neighbors(b1).filter(|&a2| {
                        a1 != a2
                            && node_qualifies(a2)
                            && graph.has_undir_edge(a2, b2)
                            && !graph.has_undir_edge(a1, a2)
                    }) {
                        let mut na_union = compute_neighbor_union(a1, a2);
                        na_union.swap_remove(na_union.iter().position(|&x| x == b1).unwrap());
                        na_union.swap_remove(na_union.iter().position(|&x| x == b2).unwrap());

                        if na_union.len() > 2 {
                            continue;
                        }

                        if na_union.iter().any(|a| nb_union.iter().any(|b| b == a)) {
                            continue;
                        }

                        nb_union.swap_remove(nb_union.iter().position(|&x| x == a2).unwrap());

                        return Some((a1, a2, b1, b2, na_union, nb_union));
                    }
                }
            }
            None
        })();

        if let Some((a1, a2, b1, b2, na, nb)) = c4 {
            for &a in &na {
                for &b in &nb {
                    graph.try_add_edge(a, b);
                    graph.try_add_edge(b, a);
                }
            }

            graph.remove_edges_of_nodes(&[a1, a2, b1, b2]);

            debug_assert!(!na.is_empty() && !nb.is_empty()); // no possible as all nodes in the cycle have degree >= 3

            postprocessor.push(Box::new(PostRuleDesk {
                search: (na[0], if na.len() > 1 { Some(na[1]) } else { None }),
                if_found: (b1, b2),
                if_not_found: (a1, a2),
            }));

            true
        } else {
            false
        }
    })
}

#[derive(Debug)]
struct PostRuleDesk {
    search: (Node, Option<Node>),
    if_found: (Node, Node),
    if_not_found: (Node, Node),
}

impl PostprocessorRule for PostRuleDesk {
    fn apply(&mut self, fvs: &mut Vec<Node>) {
        if fvs.contains(&self.search.0) && self.search.1.map_or(true, |x| fvs.contains(&x)) {
            fvs.push(self.if_found.0);
            fvs.push(self.if_found.1);
        } else {
            fvs.push(self.if_not_found.0);
            fvs.push(self.if_not_found.1);
        }
    }

    fn will_add(&self) -> Node {
        2
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This rule is due to Hespe et al. "WeGotYouCovered: The Winning Solver from the PACE 2019
/// Implementation Challenge, Vertex Cover Track" (section 4.1 / "alternative")
///
/// We are searching for a nodes u and v such that the neighborhood Q=N(v) \ {u} is a complete graph.
/// Then we can remove nodes {u}, {v} and the undirected neighbors shared. In the second stage, we
/// will either add {u} or {v} to the DFVS depending on whose neighborhood is already completely in
/// the solution.
pub fn apply_rule_funnel<G: ReducibleGraph>(
    graph: &mut G,
    dfvs: &mut Vec<Node>,
    postprocessor: &mut Postprocessor,
) -> bool {
    repeat_while(|| {
        let uv = graph
            .vertices()
            .filter(|&v| !graph.has_dir_edges(v) && graph.undir_degree(v) > 2)
            .find_map(|u| {
                let req_clique_degree = graph.undir_degree(u); // required clique size - 1
                graph
                    .undir_neighbors(u)
                    .filter(|&u| !graph.has_dir_edges(u))
                    .find_map(|v| {
                        if graph.undir_neighbors(v).filter(|&w| w != u).any(|w| {
                            graph.undir_degree(w) < req_clique_degree || graph.has_dir_edges(w)
                        }) || graph.undir_neighbors(v).filter(|&w| w != u).any(|w| {
                            graph
                                .undir_neighbors(v)
                                .filter(|&x| x != u && x < w)
                                .any(|x| !graph.has_undir_edge(w, x))
                        }) {
                            None
                        } else {
                            Some((u, v))
                        }
                    })
            });

        if let Some((u, v)) = uv {
            let common = graph
                .undir_neighbors(u)
                .filter(|&w| graph.has_undir_edge(v, w))
                .collect_vec();

            dfvs.extend(&common);

            let nu_without_common = graph
                .undir_neighbors(u)
                .filter(|w| *w != v && !common.contains(w))
                .collect_vec();
            let nv_without_common = graph
                .undir_neighbors(v)
                .filter(|w| *w != u && !common.contains(w))
                .collect_vec();

            for &a in &nu_without_common {
                for &b in &nv_without_common {
                    graph.try_add_edge(a, b);
                    graph.try_add_edge(b, a);
                }
            }

            graph.remove_edges_of_nodes(&[u, v]);
            graph.remove_edges_of_nodes(&common);

            postprocessor.push(Box::new(
                if nu_without_common.len() < nv_without_common.len() {
                    PostRuleFunnel {
                        search: nu_without_common,
                        if_found: v,
                        if_not_found: u,
                    }
                } else {
                    PostRuleFunnel {
                        search: nv_without_common,
                        if_found: u,
                        if_not_found: v,
                    }
                },
            ));

            true
        } else {
            false
        }
    })
}

#[derive(Debug)]
struct PostRuleFunnel {
    search: Vec<Node>,
    if_found: Node,
    if_not_found: Node,
}

impl PostprocessorRule for PostRuleFunnel {
    fn apply(&mut self, fvs: &mut Vec<Node>) {
        fvs.push(if self.search.iter().all(|w| fvs.contains(w)) {
            self.if_found
        } else {
            self.if_not_found
        });
    }

    fn will_add(&self) -> Node {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::exact::branch_and_bound_matrix::branch_and_bound_matrix;
    use crate::graph::generators::GeneratorSubstructures;
    use crate::kernelization::tests::for_each_stress_graph_with_opt_sol;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    #[test]
    fn rule_undir_degree_two() {
        // only one two path 0 <=> 1 <=> 2
        let graph = AdjArrayUndir::from(&[(0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (4, 2)]);

        {
            let mut graph = graph.clone();
            let mut post_proc = Postprocessor::new();
            assert!(!apply_rule_undir_degree_two(&mut graph, &mut post_proc)); // cannot fold as 0 and 1 have directed edges
            assert!(post_proc.is_empty());
        }

        {
            // we cannot clone the post processor; so compute it from scratch each time
            let compute_post_proc = || {
                let mut graph = graph.clone();
                graph.remove_edge(0, 3);

                let mut post_proc = Postprocessor::new();
                assert!(apply_rule_undir_degree_two(&mut graph, &mut post_proc));

                assert_eq!(post_proc.will_add(), 1);
                assert!(!post_proc.is_empty());

                assert_eq!(graph.number_of_edges(), 1);

                post_proc
            };

            {
                // not part
                let mut post_proc = compute_post_proc();
                let mut sol = vec![];
                post_proc.finalize_with_global_solution(&mut sol);
                assert_eq!(sol, [1]);
            }

            {
                // not part
                let mut post_proc = compute_post_proc();
                let mut sol = vec![1000];
                post_proc.finalize_with_global_solution(&mut sol);
                sol.sort_unstable();
                assert_eq!(sol, [1, 1000]);
                assert!(post_proc.is_empty());
            }

            {
                // is part
                let mut post_proc = compute_post_proc();
                let folded_into = if graph.has_edge(4, 0) { 0 } else { 2 };
                let mut sol = vec![folded_into, 1000];
                post_proc.finalize_with_global_solution(&mut sol);
                sol.sort_unstable();
                assert_eq!(sol, vec![0, 2, 1000]);
                assert!(post_proc.is_empty());
            }
        }
    }

    #[test]
    fn rule_twins_degree_three() {
        // twins 0 and 1; neighbors 2,3,4; two neighborhood 5,6
        let graph = AdjArrayUndir::from(&[
            (0, 2),
            (0, 3),
            (0, 4),
            (2, 0),
            (3, 0),
            (4, 0), //
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (3, 1),
            (4, 1), //
            (5, 2),
            (2, 5),
            (5, 3),
            (3, 5),
            (6, 3),
            (3, 6),
            (6, 4),
            (4, 6),
        ]);

        let mut gen = Pcg64::seed_from_u64(1234);

        for _ in 0..10 {
            // cannot find twins as one edge is missing
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();
            let mut dfvs = Vec::new();

            loop {
                let u = gen.gen_range(0..5);
                let v = gen.gen_range(0..5);
                if graph.try_remove_edge(u, v) {
                    break;
                }
            }

            let m = graph.number_of_edges();
            assert!(!apply_rule_twins_degree_three(
                &mut graph, &mut dfvs, &mut pp
            ));
            assert_eq!(graph.number_of_edges(), m);
        }

        {
            // found twins, but cannot decide yet
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();
            let mut dfvs = Vec::new();
            assert!(apply_rule_twins_degree_three(
                &mut graph, &mut dfvs, &mut pp
            ));
            assert_eq!(pp.will_add(), 2);
            assert_eq!(graph.number_of_edges(), 4);
            assert!(dfvs.is_empty());

            pp.finalize_with_global_solution(&mut dfvs);
            dfvs.sort_unstable();
            assert_eq!(dfvs, vec![0, 1]);
        }

        {
            // found twins, but cannot decide yet
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();
            let mut dfvs = Vec::new();
            assert!(apply_rule_twins_degree_three(
                &mut graph, &mut dfvs, &mut pp
            ));
            assert_eq!(pp.will_add(), 2);
            assert_eq!(graph.number_of_edges(), 4);
            assert!(dfvs.is_empty());

            let gadget = (0..5)
                .into_iter()
                .find(|&u| graph.undir_degree(u) > 0)
                .unwrap();

            dfvs.push(gadget);

            pp.finalize_with_global_solution(&mut dfvs);
            dfvs.sort_unstable();
            assert_eq!(dfvs, vec![2, 3, 4]);
        }

        for i in 0..10 {
            // find twins with cycle in neighborhood
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();
            let mut dfvs = Vec::new();

            if i == 0 {
                graph.add_edge(2, 3);
                graph.add_edge(3, 4);
                graph.add_edge(4, 2);
            } else if i == 1 {
                graph.add_edge(2, 3);
                graph.add_edge(3, 4);
                graph.add_edge(4, 2);
            } else {
                let u = gen.gen_range(2..5);
                let v = loop {
                    let v = gen.gen_range(2..5);
                    if v != u {
                        break v;
                    }
                };
                graph.add_edge(u, v);
                graph.add_edge(v, u);
            }

            assert!(apply_rule_twins_degree_three(
                &mut graph, &mut dfvs, &mut pp
            ));
            dfvs.sort_unstable();
            assert_eq!(dfvs, [2, 3, 4]);
            assert_eq!(graph.number_of_edges(), 0);
        }
    }

    #[test]
    fn rule_desk() {
        // cycle 0..=3, neighbors (4, 5), (6)
        let mut graph = AdjArrayUndir::from(&[
            (0, 4),
            (4, 0), // a1
            (1, 6),
            (6, 1), // b1
            (2, 5),
            (5, 2), // a2
            (3, 6),
            (6, 3), // b2
        ]);
        graph.connect_cycle([0, 1, 2, 3]);
        graph.connect_cycle([3, 2, 1, 0]);

        let mut gen = Pcg64::seed_from_u64(1234);

        for _ in 0..10 {
            // cannot desk as one edge is missing
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();

            loop {
                let u = gen.gen_range(0..4);
                let v = gen.gen_range(0..4);
                if graph.try_remove_edge(u, v) {
                    break;
                }
            }

            let m = graph.number_of_edges();
            assert!(!apply_rule_desk(&mut graph, &mut pp));
            assert_eq!(graph.number_of_edges(), m);
        }

        {
            // found desk
            let mut graph = graph.clone();
            let mut pp = Postprocessor::new();

            assert!(apply_rule_desk(&mut graph, &mut pp));
            assert_eq!(pp.will_add(), 2);
            assert_eq!(graph.number_of_edges(), 4);
        }
    }

    #[test]
    fn rule_funnel() {
        let mut graph = AdjArrayUndir::new(7);
        graph.add_edges(&[(0, 1), (1, 0), (1, 4), (4, 1)]);
        graph.connect_nodes(
            &BitSet::new_all_unset_but(8, [0u32, 2, 3, 4, 6].iter().copied()),
            false,
        );

        let mut pp = Postprocessor::new();
        let mut dfvs = Vec::new();

        assert!(apply_rule_funnel(&mut graph, &mut dfvs, &mut pp));
        assert_eq!(pp.will_add(), 1);
        assert_eq!(dfvs, vec![4]);

        pp.finalize_with_global_solution(&mut dfvs);
        dfvs.sort_unstable();
        assert_eq!(dfvs, vec![0, 4]);
    }

    pub(super) fn stress_test_two_staged_kernel<
        F: Fn(&mut AdjArrayUndir, &mut Vec<Node>, &mut Postprocessor) -> bool,
    >(
        kernel: F,
    ) {
        let mut num_applied = 0;
        for_each_stress_graph_with_opt_sol(|filename, org_graph, opt_sol| {
            let mut graph = org_graph.clone();
            let digest_before = graph.digest_sha256();

            let mut rule_fvs = Vec::new();
            let mut pp = Postprocessor::new();
            let applied = kernel(&mut graph, &mut rule_fvs, &mut pp);
            num_applied += applied as usize;
            let digest_after = graph.digest_sha256();

            if digest_before == digest_after {
                assert!(!applied, "File: {}", filename);
                assert!(rule_fvs.is_empty(), "File: {}", filename);
                assert_eq!(pp.will_add(), 0, "File: {}", filename);
                return;
            }

            assert!(applied, "File: {}", filename);

            let mut kernel_fvs = branch_and_bound_matrix(&graph, None).unwrap();
            kernel_fvs.extend(&rule_fvs);
            pp.finalize_with_global_solution(&mut kernel_fvs);

            assert_eq!(kernel_fvs.len(), opt_sol as usize, "File: {}", filename);
            assert!(
                org_graph
                    .vertex_induced(&BitSet::new_all_set_but(
                        graph.len(),
                        kernel_fvs.iter().copied()
                    ))
                    .0
                    .is_acyclic(),
                "File: {}",
                filename
            );
        });

        if num_applied == 0 {
            println!("Rule was never applied");
        }
    }

    #[test]
    fn stress_desk() {
        stress_test_two_staged_kernel(|g, _, pp| apply_rule_desk(g, pp));
    }

    #[test]
    fn stress_undir_degree_two() {
        stress_test_two_staged_kernel(|g, _, pp| apply_rule_undir_degree_two(g, pp));
    }

    #[test]
    fn stress_funnel() {
        stress_test_two_staged_kernel(apply_rule_funnel);
    }

    #[test]
    fn stress_twins_degree_three() {
        stress_test_two_staged_kernel(apply_rule_twins_degree_three);
    }
}
