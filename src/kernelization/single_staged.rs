use super::*;
use fxhash::FxHashSet;
use std::collections::HashSet;
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

        let neighbors: HashSet<Node, _> = FxHashSet::from_iter(graph.undir_neighbors(u));

        if neighbors.iter().any(|&v| {
            graph
                .undir_neighbors(v)
                .filter(|x| *x == u || neighbors.contains(x))
                .take(k as usize)
                .count()
                < k as usize
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
    let mut outer_applied = false;
    let mut num_nodes = graph
        .vertices()
        .filter(|&u| graph.total_degree(u) > 0)
        .count() as Node;

    loop {
        let mut applied = false;
        for node in graph.vertices_range() {
            if graph.out_degree(node) == graph.in_degree(node)
                && graph.out_degree(node) == (num_nodes - (!graph.has_edge(node, node)) as Node)
                && !graph.has_edge(node, node)
            {
                fvs.push(node);
                graph.remove_edges_at_node(node);
                applied = true;
                num_nodes -= 1;
            }
        }
        outer_applied |= applied;
        if !applied {
            return outer_applied;
        }
    }
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
    fn complete_node() {
        let mut pre_processor = {
            let graph =
                AdjArrayUndir::from(&[(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 3), (2, 2)]);
            PreprocessorReduction::from(graph)
        };

        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(0, 0);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(0, 3);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(3, 0);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.remove_edge(0, 0);
        assert!(pre_processor.apply_complete_node());
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
        test_pre_process.apply_pie_reduction();
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
        test_pre_process.apply_pie_reduction();
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
        test_pre_process.apply_dome_reduction();
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
}
