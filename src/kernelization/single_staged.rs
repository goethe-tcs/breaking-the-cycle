use super::*;
use fxhash::FxHashSet;
use itertools::Itertools;
use std::iter::FromIterator;

/// rule 1 - self-loop
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_1<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.has_edge(u, u) {
            fvs.push(u);
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }
    applied
}

/// rule 3 sink/source nodes
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_3<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if (graph.in_degree(u) == 0) != (graph.out_degree(u) == 0) {
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }

    applied
}

/// rule 4 chaining nodes with deleting self loop
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_4<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.in_degree(u) == 1 || graph.out_degree(u) == 1 {
            debug_assert!(!graph.has_edge(u, u));
            let loops = graph.contract_node(u);
            fvs.extend(&loops);
            for v in loops {
                graph.remove_edges_at_node(v);
            }
            applied = true;
        }
    }
    applied
}

/// Safe Di-Cliques Reduction, requires self loops to be deleted
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_di_cliques<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        let k = graph.undir_degree(u);

        if k == 0 || graph.undir_neighbors(u).any(|v| graph.undir_degree(v) < k) {
            continue;
        }

        let neighbors = FxHashSet::from_iter(graph.undir_neighbors(u));

        if neighbors.iter().any(|&v| {
            graph
                .undir_neighbors(v)
                .filter(|x| *x == u || neighbors.contains(x))
                .take(k as usize)
                .count()
                != k as usize
        }) {
            continue;
        }

        // if u is only in 1 clique we check here if we can safely delete all nodes but u from that
        // clique. Either u has only out- or in-going edges or there is no circle back to u
        // outside of the clique. This last point is checked via breadth first search
        if graph.in_degree(u) == k
            || graph.out_degree(u) == k
            || !graph.is_node_on_cycle_after_deleting(u, neighbors.iter().copied())
        {
            for &node in &neighbors {
                graph.remove_edges_at_node(node);
            }
            fvs.extend(neighbors.iter());
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }
    applied
}

pub fn apply_rule_complete_node<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut num_nodes = graph
        .vertices()
        .filter(|&u| graph.total_degree(u) > 0)
        .count() as Node;

    apply_rule_1(graph, fvs)
        | repeat_while(|| {
            if num_nodes <= 1 {
                return false;
            }

            let mut applied = false;
            for node in graph.vertices_range() {
                if num_nodes > 1 && graph.undir_degree(node) == num_nodes - 1 {
                    fvs.push(node);
                    graph.remove_edges_at_node(node);
                    applied = true;
                    num_nodes -= 1;

                    if num_nodes == 1 {
                        return true;
                    }
                }
            }
            applied
        })
}

/// PIE reduction rule
///
/// Looks for directed edges that are not strongly connected after removing undirected edges
pub fn apply_rule_pie<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    let mut graph_minus_pie = AdjArray::new(graph.len());
    // get directed out_neighbors for every node and create a graph with only directed edges
    for node in graph.vertices_range() {
        for out_neighbor in graph.out_only_neighbors(node) {
            graph_minus_pie.add_edge(node, out_neighbor);
        }
    }
    let graph_sccs = graph_minus_pie.partition_into_strongly_connected_components();
    if graph_sccs.number_of_classes() == 1 && graph_sccs.number_of_unassigned() == 0 {
        return applied;
    }
    // check for every edge of our graph without undirected edges if the nodes have
    // stayed in the same scc, if not we can delete the directed edge
    for (u, v) in graph_minus_pie.edges_iter() {
        if graph_sccs.class_of_node(u) == None
            || graph_sccs.class_of_node(v) == None
            || graph_sccs.class_of_node(u) != graph_sccs.class_of_node(v)
        {
            graph.remove_edge(u, v);
            applied = true;
        }
    }
    applied
}

/// DOME reduction rule
///
/// Looks for directed edges that get dominated by other edges (un/directed edges)
pub fn apply_rule_dome<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        loop {
            if graph.in_degree(u) == 0 || graph.out_degree(u) == 0 {
                break;
            }

            let in_only_neigh_u = graph.in_only_neighbors(u).collect::<FxHashSet<_>>();
            let out_neigh_u = graph.out_neighbors(u).collect::<FxHashSet<_>>();

            let v = graph.out_only_neighbors(u).find(|&neighbor|
                // test if in_only_neigh_u is subset of in_only_neighbors(neighbor)
                graph
                    .in_neighbors(neighbor)
                    .filter(|v| in_only_neigh_u.contains(v))
                    .take(in_only_neigh_u.len())
                    .count()
                    == in_only_neigh_u.len()
                    // test if out_only_neighbors(neighbor) is subset of out_neigh_u
                    || graph
                    .out_only_neighbors(neighbor)
                    .all(|v| out_neigh_u.contains(&v)));

            if let Some(v) = v {
                applied = true;
                graph.remove_edge(u, v);
            } else {
                break;
            }
        }
    }
    applied
}

pub fn apply_rule_c4<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    /* We are looking for an "undirected" cycle of four nodes (not induced; i.e. there
      may be additional edges).
           u
         /   \
       v       w
         \   /
           x

      In such a structure, we have to delete at least {u, x} or {v, w}. Thus, the remaining
      pair needs to have (amongst others) an undir-degree of exactly 2. To prune the search
      space we require that the undir-degree of u *is* 2 --- {v,w} may have larger undir-degree.

      To make the deletion of nodes (and adding to the DFVS) compatible with a globally optimal
      solution,  we require that the remaining pair not to have cycles outside the subgraph. Thus,
      we can delete all four nodes and add two to the DFVS.
    */

    let mut applied = 0;

    loop {
        let c4 = (|| {
            'foru: for u in graph.vertices_range() {
                if graph.undir_degree(u) != 2 {
                    continue;
                }

                let (v, dv, w, dw) = {
                    // v has smaller/eq degree
                    let mut neighs = graph.undir_neighbors(u);
                    let v = neighs.next().unwrap();
                    let w = neighs.next().unwrap();
                    debug_assert!(neighs.next().is_none());
                    debug_assert_ne!(u, v);
                    debug_assert_ne!(u, w);
                    debug_assert_ne!(v, w);

                    let dv = graph.undir_degree(v);
                    let dw = graph.undir_degree(v);
                    if dv < dw {
                        (v, dv, w, dw)
                    } else {
                        (w, dw, v, dv)
                    }
                };

                if dv < 2 {
                    continue;
                }

                // if 2 <= max(dw, dv) = dw, we have to delete the {v, w} pair since at least w
                // will keep a cycle outside of the C4 if we were to delete {u, x}.
                let has_to_delete_vw = dw > 2;
                let mut u_acyclic_without_vw = None;

                for x in graph.undir_neighbors(v) {
                    if x == u
                        || x == v
                        || x == w
                        || (has_to_delete_vw && graph.undir_degree(x) > 2)
                        || !graph.has_edge(w, x)
                        || !graph.has_edge(x, w)
                    {
                        continue;
                    }

                    if u_acyclic_without_vw.is_none() {
                        u_acyclic_without_vw =
                            Some(!graph.is_node_on_cycle_after_deleting(u, [v, w]));

                        if has_to_delete_vw && !u_acyclic_without_vw.unwrap() {
                            continue 'foru;
                        }
                    }

                    if u_acyclic_without_vw.unwrap()
                        && !graph.is_node_on_cycle_after_deleting(x, [v, w])
                    {
                        return Some(((v, w), (u, x)));
                    }

                    if !has_to_delete_vw
                        && !graph.is_node_on_cycle_after_deleting(v, [u, x])
                        && !graph.is_node_on_cycle_after_deleting(w, [u, x])
                    {
                        return Some(((u, x), (v, w)));
                    }
                }
            }
            None
        })();

        if let Some(((del1, del2), (keep1, keep2))) = c4 {
            graph.remove_edges_at_node(del1);
            graph.remove_edges_at_node(del2);
            graph.remove_edges_at_node(keep1);
            graph.remove_edges_at_node(keep2);
            fvs.push(del1);
            fvs.push(del2);

            applied += 1;
        } else {
            return applied > 0;
        }
    }
}

/// Removes unconfined vertices as proposed by Kneis et al. in "A fine-grained analysis of a simple
/// independent set algorithm."  We implement the algorithm based on Akiba and Iwata "Branch-and-reduce
/// exponential/FPT algorithms in practice: A case study of vertex cover". Some adoption were necessary
/// to make it compatible with the DFVS problem (we basically restrict ourselves to the undirected subgraph)
pub fn apply_rule_unconfined<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;

    // capacities are estimated from profiling a couple of runs
    let mut set_s = Vec::with_capacity(8);
    let mut neighbors_of_s = Vec::with_capacity(32);

    loop {
        let mut local_applied = false;
        for v in graph.vertices_range() {
            if !graph.has_undir_edges(v) || graph.has_dir_edges(v) {
                continue;
            }

            set_s.clear();
            neighbors_of_s.clear();

            let add_to_s = |v: Node, set_s: &mut Vec<Node>, neighbors_of_s: &mut Vec<Node>| {
                neighbors_of_s.extend(graph.undir_neighbors(v).filter(|w| !set_s.contains(w)));
                if let Some(i) = neighbors_of_s.iter().position(|&x| x == v) {
                    neighbors_of_s.swap_remove(i);
                }
                set_s.push(v)
            };

            add_to_s(v, &mut set_s, &mut neighbors_of_s);

            let is_unconfined = loop {
                let u = neighbors_of_s
                    .iter()
                    .filter_map(|&u| {
                        if graph
                            .undir_neighbors(u)
                            .filter(|v| set_s.contains(v))
                            .take(2)
                            .count()
                            == 1
                        {
                            let mut set = graph
                                .undir_neighbors(u as Node)
                                .filter(|v| !neighbors_of_s.contains(v) && !set_s.contains(v));

                            // try to extract the first entry and count the remainder
                            let w = set.next();
                            Some((w.is_some() as usize + set.count(), w))
                        } else {
                            None
                        }
                    })
                    .min_by_key(|&(n, _)| n);

                if let Some((n, w)) = u {
                    if n == 0 {
                        break neighbors_of_s.iter().all(|&v| !graph.has_dir_edges(v));
                    }

                    if n == 1 {
                        let w = w.unwrap();
                        if graph.has_dir_edges(w) {
                            break false;
                        }
                        add_to_s(w, &mut set_s, &mut neighbors_of_s);
                        continue;
                    }
                }

                break false;
            };

            if is_unconfined {
                graph.remove_edges_at_node(v);
                fvs.push(v);
                local_applied = true;
            }
        }

        if !local_applied {
            return applied;
        }

        applied = true;
    }
}

pub fn apply_rule_domn<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;

    loop {
        let node = graph.vertices().find(|&u| {
            graph.in_degree(u) > 0
                && graph.out_degree(u) > 0
                && graph.in_neighbors(u).all(|i| {
                    graph
                        .out_neighbors(u)
                        .all(|j| i == j || graph.has_edge(i, j))
                })
        });

        if let Some(node) = node {
            applied = true;
            let loops = graph.undir_neighbors(node).collect_vec();

            graph.remove_edges_at_node(node);
            graph.remove_edges_of_nodes(&loops);

            fvs.extend(&loops);
        } else {
            return applied;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_1() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_1();

        assert_eq!(test_pre_process.graph.number_of_edges(), 4);
        assert_eq!(test_pre_process.in_fvs.len(), 2);
    }

    #[test]
    fn rule_3() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_3();
        assert_eq!(test_pre_process.graph.number_of_edges(), 8);
        assert_eq!(test_pre_process.graph.out_degree(5), 0);
    }

    #[test]
    fn rule_4_neighbor_is_neighbor() {
        let mut test_pre_process = {
            let graph = AdjArrayUndir::from(&[
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 0),
                (2, 1),
                (2, 3),
                (2, 5),
                (3, 2),
                (3, 1),
                (3, 4),
                (4, 1),
                (4, 3),
                (4, 5),
                (5, 2),
                (5, 3),
            ]);

            let test_pre_process = PreprocessorReduction::from(graph);
            test_pre_process
        };
        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 3);
        assert_eq!(test_pre_process.graph.number_of_edges(), 0);
    }

    #[test]
    fn rule_4_neighbor_not_neighbor() {
        let graph = AdjArrayUndir::from(&[
            (0, 2),
            (0, 5),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 5),
            (3, 2),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 5),
            (5, 2),
            (5, 3),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 0);
        assert_eq!(test_pre_process.graph.number_of_edges(), 10);
    }

    #[test]
    fn pie_reduction() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 4),
            (2, 5),
            (3, 0),
            (3, 4),
            (4, 1),
            (4, 5),
            (4, 2),
            (5, 3),
            (5, 0),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 16);
        test_pre_process.apply_rule_pie();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 14);
    }

    #[test]
    fn pie_reduction_nones() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 0),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 8);
        test_pre_process.apply_rule_pie();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 6);
    }

    #[test]
    fn dome_reduction() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 5),
            (4, 3),
            (4, 5),
            (5, 0),
            (5, 4),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 13);
        test_pre_process.apply_rule_dome();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 9);
    }

    fn create_test_pre_processor() -> PreprocessorReduction<AdjArrayUndir> {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (1, 4),
            (2, 2),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 3),
            (4, 0),
            (5, 4),
        ]);

        let test_pre_process = PreprocessorReduction::from(graph);
        test_pre_process
    }

    #[test]
    fn unconfined() {
        let mut graph = AdjArrayUndir::from(&[(0, 1), (1, 0)]);
        let mut fvs = Vec::new();
        assert!(apply_rule_unconfined(&mut graph, &mut fvs));
    }
}
