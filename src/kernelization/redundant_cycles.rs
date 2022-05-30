use super::*;
use fxhash::{FxBuildHasher, FxHashSet};
use itertools::Itertools;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::ops::Neg;

/// This rule search for small cycles of length 2 (aka undirected edges), 3 and 4. Given some cycle
/// C, if there exists an edge outside of C that is not part of any cycle if ANY of C's nodes is
/// deleted, we can delete that edge safely (since we know that we have to delete at least one
/// of C's nodes eventually).
pub fn apply_rule_redundant_cycle<G: ReducibleGraph>(graph: &mut G) -> bool {
    // todo: this implementation is a proof-of-concept and should REALLY be rewritten cleaner.

    let org_num_edges = graph.number_of_edges();
    let graph_minus_pie = graph_without_pie(graph);

    // get directed out_neighbors for every node and create a graph with only directed edges
    let full_sccs = graph_minus_pie
        .partition_into_strongly_connected_components()
        .split_into_subgraphs(graph);

    for (full_scc, mapping) in full_sccs {
        let mut dir_scc = graph_without_pie(&full_scc);

        let mut memory = vec![None; dir_scc.len()];

        fn edges_without_node<G: ReducibleGraph>(
            memory: &mut [Option<HashSet<Edge, FxBuildHasher>>],
            graph: &G,
            u: Node,
        ) -> HashSet<Edge, FxBuildHasher> {
            if let Some(res) = memory[u as usize].as_ref() {
                return res.clone();
            }

            let mut without_node = graph.clone();
            without_node.remove_edges_at_node(u);

            while apply_rule_3(&mut without_node) {}

            let mut res = FxHashSet::from_iter(graph.edges_iter());
            for x in without_node.edges_iter() {
                res.remove(&x);
            }

            memory[u as usize] = Some(res);
            memory[u as usize].as_ref().unwrap().clone()
        }

        let mut verts = full_scc.vertices().collect_vec();
        verts.sort_by_key(|&u| (dir_scc.total_degree(u) as i64).neg());

        for u in verts {
            if full_scc
                .undir_neighbors(u)
                .all(|v| u > v || !dir_scc.has_dir_edges(v))
            {
                continue;
            }

            let without_u = edges_without_node(&mut memory, &dir_scc, u);

            let found = full_scc
                .undir_neighbors(u)
                .filter(|&v| u < v && dir_scc.has_dir_edges(v))
                .map(|v| {
                    let without_v = edges_without_node(&mut memory, &dir_scc, v);
                    (v, without_v.intersection(&without_u).copied().collect_vec())
                })
                .collect_vec();

            if found.is_empty() {
                continue;
            }

            let mut redundant = found
                .iter()
                .flat_map(|(_, vec)| vec.iter().copied())
                .collect_vec();
            redundant.sort_unstable();
            redundant.dedup();

            for (a, b) in redundant {
                dir_scc.try_remove_edge(a, b);
                memory[u as usize].as_mut().unwrap().remove(&(a, b));

                for (v, _) in &found {
                    memory[*v as usize].as_mut().unwrap().remove(&(a, b));
                }

                graph.try_remove_edge(mapping.old_id_of(a).unwrap(), mapping.old_id_of(b).unwrap());
            }
        }

        loop {
            if dir_scc.number_of_edges() <= 3 {
                break;
            }

            let red = dir_scc.vertices_range().find_map(|u| {
                let without_u = edges_without_node(&mut memory, &dir_scc, u);
                if without_u.len() <= 3 {
                    return None;
                }

                dir_scc.out_neighbors(u).filter(|&v| v > u).find_map(|v| {
                    if dir_scc.out_neighbors(v).all(|v| v <= u) {
                        return None;
                    }

                    let without_v = edges_without_node(&mut memory, &dir_scc, v);
                    let mut inter1: HashSet<Edge, FxBuildHasher> =
                        without_u.intersection(&without_v).copied().collect();

                    if inter1.len() <= 3 {
                        return None;
                    }

                    inter1.remove(&(u, v));

                    dir_scc.out_neighbors(v).filter(|&v| v > u).find_map(|w| {
                        if !dir_scc.has_edge(w, u) {
                            return None;
                        }

                        let mut without_w = edges_without_node(&mut memory, &dir_scc, w);
                        without_w.remove(&(v, w));
                        without_w.remove(&(w, u));

                        let inter = without_w.intersection(&inter1).copied().collect_vec();

                        if inter.is_empty() {
                            None
                        } else {
                            Some((u, v, w, inter))
                        }
                    })
                })
            });

            if let Some((u, v, w, redundant)) = red {
                //info!("3Redundant: {:?}", &redundant);

                for (a, b) in redundant {
                    dir_scc.try_remove_edge(a, b);
                    graph.try_remove_edge(
                        mapping.old_id_of(a).unwrap(),
                        mapping.old_id_of(b).unwrap(),
                    );
                    memory[u as usize].as_mut().unwrap().remove(&(a, b));
                    memory[v as usize].as_mut().unwrap().remove(&(a, b));
                    memory[w as usize].as_mut().unwrap().remove(&(a, b));
                }
            } else {
                break;
            }
        }

        loop {
            if dir_scc.number_of_edges() <= 4 {
                break;
            }

            let red = dir_scc.vertices_range().find_map(|u| {
                let without_u = edges_without_node(&mut memory, &dir_scc, u);
                if without_u.len() <= 4 {
                    return None;
                }

                dir_scc.out_neighbors(u).filter(|&v| v > u).find_map(|v| {
                    if dir_scc.out_neighbors(v).all(|v| v <= u) {
                        return None;
                    }

                    let without_v = edges_without_node(&mut memory, &dir_scc, v);
                    let mut inter1: HashSet<Edge, FxBuildHasher> =
                        without_u.intersection(&without_v).copied().collect();

                    if inter1.len() <= 4 {
                        return None;
                    }

                    inter1.remove(&(u, v));

                    dir_scc.out_neighbors(v).filter(|&v| v > u).find_map(|w| {
                        if dir_scc.has_edge(w, u) {
                            // 3 cycles have already been dealt with
                            return None;
                        }

                        if dir_scc.out_neighbors(w).all(|v| v <= u) {
                            return None;
                        }

                        let without_w = edges_without_node(&mut memory, &dir_scc, w);
                        let mut inter2: HashSet<Edge, FxBuildHasher> =
                            without_w.intersection(&inter1).copied().collect();

                        if inter2.len() <= 4 {
                            return None;
                        }
                        inter2.remove(&(v, w));

                        dir_scc.out_neighbors(w).filter(|&v| v > u).find_map(|x| {
                            if x == u || !dir_scc.has_edge(x, u) {
                                return None;
                            }

                            debug_assert!(u != v && u != w && u != x && v != w && v != x && w != x);

                            let mut without_x = edges_without_node(&mut memory, &dir_scc, x);
                            without_x.remove(&(w, x));
                            without_x.remove(&(x, u));

                            let inter = without_x.intersection(&inter2).copied().collect_vec();

                            if inter.is_empty() {
                                None
                            } else {
                                Some((u, v, w, x, inter))
                            }
                        })
                    })
                })
            });

            if let Some((u, v, w, x, redundant)) = red {
                //info!("4Redundant: {:?}", &redundant);

                for (a, b) in redundant {
                    dir_scc.try_remove_edge(a, b);
                    graph.try_remove_edge(
                        mapping.old_id_of(a).unwrap(),
                        mapping.old_id_of(b).unwrap(),
                    );
                    memory[u as usize].as_mut().unwrap().remove(&(a, b));
                    memory[v as usize].as_mut().unwrap().remove(&(a, b));
                    memory[w as usize].as_mut().unwrap().remove(&(a, b));
                    memory[x as usize].as_mut().unwrap().remove(&(a, b));
                }

                continue;
            }

            break;
        }
    }

    org_num_edges != graph.number_of_edges()
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn stress_redundant_cycles() {
        stress_test_kernel(|graph, _, _| Some(apply_rule_redundant_cycle(graph)));
    }
}
