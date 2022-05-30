use super::*;
use crate::bitset::BitSet;
use crate::graph::network_flow::{EdmondsKarp, ResidualBitMatrix, ResidualNetwork};

/// Converts a graph into a Vector of BitSets. But every node v of the graph is split into two nodes.
/// One node has the ingoing edges of v and one has the outgoing edges of v.
/// Also adds an edge from the node with ingoing edges to the node with outgoing edges.
fn create_capacity_for_many_petals<G: ReducibleGraph>(graph: &mut G) -> Vec<BitSet> {
    let graph_len_double = graph.len() * 2;
    let mut capacity = vec![BitSet::new(graph_len_double); graph_len_double];
    for node in graph.vertices_range() {
        capacity[node as usize].set_bit(node as usize + graph.len());
        for out_node in graph.out_neighbors(node) {
            capacity[node as usize + graph.len()].set_bit(out_node as usize);
        }
    }
    capacity
}

/// Updates capacity and graph.
fn perform_petal_reduction_rule_5<G: ReducibleGraph>(
    graph: &mut G,
    node: Node,
    capacity: &mut [BitSet],
    fvs: &mut Vec<Node>,
) {
    // removing edges from capacity
    remove_edges_at_capacity_node(capacity, node, graph.number_of_nodes());
    capacity[(node + graph.len() as Node) as usize].unset_all();
    capacity[node as usize].unset_bit((node + (graph.len() as Node)) as usize);
    for bit_vector in capacity.iter_mut() {
        bit_vector.unset_bit(node as usize);
    }

    // must be collected before edges at node get removed.
    let in_neighbors: Vec<_> = graph.in_neighbors(node).collect();
    let out_neighbors: Vec<_> = graph.out_neighbors(node).collect();

    // removing edges from graph
    graph.remove_edges_at_node(node);

    // add all possible edges for (in_neighbors, out_neighbors)
    let mut loop_to_delete: Vec<Node> = vec![];
    for in_neighbor in in_neighbors {
        if out_neighbors.contains(&in_neighbor) {
            loop_to_delete.push(in_neighbor);
            continue;
        }
        for &out_neighbor in &out_neighbors {
            let edge_from = (in_neighbor as usize) + graph.len();
            let edge_to = out_neighbor as usize;
            capacity[edge_from].set_bit(edge_to); // adds new edge to capacity

            graph.try_add_edge(in_neighbor, out_neighbor); // adds new edge to graph
        }
    }
    for node in loop_to_delete {
        fvs.push(node);
        graph.remove_edges_at_node(node);
        remove_edges_at_capacity_node(capacity, node, graph.number_of_nodes());
    }
}

fn count_petals<G: ReducibleGraph>(
    node: Node,
    graph: &G,
    mut capacity: Vec<BitSet>,
    mut labels: Vec<Node>,
    count_up_to: Node,
) -> (Vec<BitSet>, Vec<Node>, usize) {
    let undir_degree = graph.undir_degree(node);
    let n = graph.len();

    if undir_degree >= count_up_to {
        return (capacity, labels, count_up_to as usize);
    }

    // prepare the capacity to run num_disjoint()
    capacity[node as usize].unset_bit(node as usize + n);

    for v in graph.undir_neighbors(node) {
        let was_set1 = capacity[node as usize + n].unset_bit(v as usize);
        let was_set2 = capacity[v as usize].unset_bit(v as usize + n);
        assert!(was_set1 && was_set2);
    }

    let petal_bit_matrix =
        ResidualBitMatrix::from_capacity_and_labels(capacity, labels, node + n as Node, node);

    let mut ec = EdmondsKarp::new(petal_bit_matrix);
    ec.set_remember_changes(true);
    let petal_count = ec.count_num_disjoint_upto(count_up_to - undir_degree) as usize;
    ec.undo_changes();

    (capacity, labels) = ec.take(); // needed because capacity and labels is moved when petal_bit_matrix is created

    for v in graph.undir_neighbors(node) {
        capacity[node as usize + n].set_bit(v as usize);
        capacity[v as usize].set_bit(v as usize + n);
    }

    (capacity, labels, petal_count + undir_degree as usize)
}

/// Checks for each node u on a given graph, if u has exactly one petal.
/// If so, the graph gets reduced. That means, the node u with petal=1 gets deleted and all possible
/// edges (in_neighbors(u), out_neighbour(u)) are added to the graph.
pub fn apply_rule_5<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;

    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        let petal_count;
        (capacity, labels, petal_count) = count_petals(node, graph, capacity, labels, 2);

        // actual reduction of graph (if petal_count = 1)
        if petal_count == 1 {
            applied = true;
            perform_petal_reduction_rule_5(graph, node as Node, &mut capacity, fvs);
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }
    applied
}

pub fn apply_rule_6<G: ReducibleGraph>(
    graph: &mut G,
    upper_bound: Node,
    fvs: &mut Vec<Node>,
) -> Option<bool> {
    let num_nodes = graph
        .vertices()
        .filter(|&u| graph.total_degree(u) > 0)
        .count() as Node;
    if num_nodes < upper_bound {
        return Some(false);
    }
    let mut applied_counter = upper_bound;
    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        if graph.in_degree(node) < applied_counter || graph.out_degree(node) < applied_counter {
            continue;
        };
        let petal_count;
        (capacity, labels, petal_count) =
            count_petals(node, graph, capacity, labels, applied_counter + 1);

        // actual reduction of graph (if petal_count > upper_bound)
        if petal_count > applied_counter as usize {
            fvs.push(node);

            // removing edges from capacity
            remove_edges_at_capacity_node(&mut capacity, node, graph.number_of_nodes());

            // removing edges from graph
            graph.remove_edges_at_node(node);
            if applied_counter == 0 {
                return None;
            }
            applied_counter -= 1;
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }

    Some(applied_counter < upper_bound)
}

pub fn apply_rules_5_and_6<G: ReducibleGraph>(
    graph: &mut G,
    upper_bound_incl: Node,
    fvs: &mut Vec<Node>,
) -> Option<bool> {
    let mut applied = false;
    let mut upper_bound_excl = upper_bound_incl + 1;

    let num_nodes = graph
        .vertices()
        .filter(|&u| graph.in_degree(u) > 0 && graph.out_degree(u) > 0)
        .take(upper_bound_excl as usize)
        .count() as Node;

    if num_nodes < upper_bound_excl
        || !graph.vertices().any(|u| {
            graph.in_degree(u) >= upper_bound_excl && graph.out_degree(u) >= upper_bound_excl
        })
    {
        return Some(apply_rule_5(graph, fvs));
    }

    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        if graph.in_degree(node) == 0 || graph.out_degree(node) == 0 {
            continue;
        }

        let petal_count = if graph.undir_degree(node) >= upper_bound_excl {
            upper_bound_excl as usize
        } else {
            let mut count_to = upper_bound_excl
                .min(graph.in_degree(node))
                .min(graph.out_degree(node));

            if count_to == 0 {
                continue;
            }

            if count_to < upper_bound_excl {
                count_to = 2;
            }

            let petal_count;
            (capacity, labels, petal_count) = count_petals(node, graph, capacity, labels, count_to);
            petal_count
        };

        if petal_count == 1 {
            let fvs_len_before = fvs.len();
            perform_petal_reduction_rule_5(graph, node as Node, &mut capacity, fvs);
            if upper_bound_excl < (fvs.len() - fvs_len_before) as Node {
                return None;
            }

            upper_bound_excl -= (fvs.len() - fvs_len_before) as Node;
            applied = true;
            continue;
        } else if petal_count >= upper_bound_excl as usize {
            fvs.push(node);

            // removing edges from capacity
            remove_edges_at_capacity_node(&mut capacity, node, graph.len() as Node);

            // removing edges from graph
            graph.remove_edges_at_node(node);
            if upper_bound_excl == 0 {
                return None;
            }
            upper_bound_excl -= 1;
            applied = true;
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }

    Some(applied)
}

fn remove_edges_at_capacity_node(capacity: &mut [BitSet], node: Node, graph_size: Node) {
    capacity[(node + graph_size) as usize].unset_all();
    capacity[node as usize].unset_bit((node + (graph_size)) as usize);
    for bit_vector in capacity.iter_mut() {
        bit_vector.unset_bit(node as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernelization::tests::stress_test_kernel;

    #[test]
    fn create_capacity_for_petals() {
        let tested_graph_len = 8;
        let one_less = (tested_graph_len - 1) as usize;
        let mut graph = create_circle(tested_graph_len);

        let capacity = create_capacity_for_many_petals(&mut graph);
        assert_eq!(capacity.len(), graph.len() * 2);
        let mut expected_edges = vec![];
        for node in 0..graph.len() - 1 {
            expected_edges.push((node, node + graph.len()));
            expected_edges.push((node + graph.len(), node + 1));
        }

        expected_edges.push((one_less, one_less + graph.len()));
        expected_edges.push((one_less + graph.len(), 0));

        for from_node in 0..capacity.len() {
            for to_node in 0..capacity.len() {
                if expected_edges.contains(&(from_node, to_node)) {
                    assert!(capacity[from_node][to_node]);
                } else {
                    assert!(!capacity[from_node][to_node]);
                }
            }
        }
    }

    #[test]
    fn petal_reduction() {
        let tested_graph_len = 4;
        let mut graph = create_circle(tested_graph_len);
        let mut capacity = create_capacity_for_many_petals(&mut graph);
        let mut fvs = vec![];
        perform_petal_reduction_rule_5(&mut graph, 0, &mut capacity, &mut fvs);
        let expected_edges_capacity = vec![(1, 5), (2, 6), (3, 7), (7, 1), (5, 2), (6, 3)];

        let expected_edges_graph = vec![(1, 2), (2, 3), (3, 1)];

        assert_eq!(0, graph.in_degree(0) + graph.out_degree(0));
        for from_node in 0..capacity.len() {
            for to_node in 0..capacity.len() {
                if expected_edges_capacity.contains(&(from_node, to_node)) {
                    assert!(capacity[from_node][to_node]);
                } else {
                    assert!(!capacity[from_node][to_node]);
                }
            }
        }
        for from_node in 0..graph.len() {
            for to_node in 0..graph.len() {
                if expected_edges_graph.contains(&(from_node, to_node)) {
                    assert!(graph.has_edge(from_node as Node, to_node as Node));
                } else {
                    assert!(!graph.has_edge(from_node as Node, to_node as Node));
                }
            }
        }
    }

    #[test]
    fn rule_5() {
        let tested_graph_len = 4;
        let graph = create_circle(tested_graph_len);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        test_pre_process.apply_rule_5();

        assert_eq!(test_pre_process.fvs(), [3]);
        assert_eq!(test_pre_process.fvs().len(), 1)
    }

    #[test]
    fn rule_6() {
        let graph = create_star(5);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert!(test_pre_process.apply_rule_6(3).unwrap());
        for node in test_pre_process.graph.vertices_range() {
            assert_eq!(test_pre_process.graph.total_degree(node), 0);
        }
        assert!(test_pre_process.in_fvs.contains(&0));

        assert!(!test_pre_process.apply_rule_6(0).unwrap());
        assert!(!test_pre_process.apply_rule_6(1).unwrap());
        assert!(!test_pre_process.apply_rule_6(0).unwrap());
        assert!(test_pre_process.in_fvs.contains(&0));
        assert_eq!(test_pre_process.in_fvs.len(), 1);
    }

    fn create_circle(graph_size: usize) -> AdjArrayUndir {
        let mut graph = AdjArrayUndir::new(graph_size);
        for i in 0..graph_size - 1 {
            graph.add_edge(i as Node, (i + 1) as Node);
        }
        graph.add_edge((graph_size - 1) as Node, 0);
        graph
    }

    /// edges go in both directions
    fn create_star(satellite_count: usize) -> AdjArrayUndir {
        let mut graph = AdjArrayUndir::new(satellite_count + 1);
        for satellite in 1..satellite_count + 1 {
            graph.add_edge(0, satellite as Node);
            graph.add_edge(satellite as Node, 0);
        }
        graph
    }

    #[test]
    fn rules_5_and_6() {
        // nodes 0 and 1 have undirected edges to fives nodes 2, .., 6
        let mut graph = AdjArrayUndir::new(7);
        for u in 2..=6 {
            graph.add_edges(&[(0, u), (u, 0), (1, u), (u, 1)]);
        }
        let digest = graph.digest_sha256();

        // ub is too high to change something
        {
            let mut graph = graph.clone();
            let mut fvs = Vec::new();
            let applied = apply_rules_5_and_6(&mut graph, 5, &mut fvs);
            assert_eq!(applied, Some(false));
            assert!(fvs.is_empty());
            assert_eq!(graph.digest_sha256(), digest);
        }

        // ub is sufficiently low that nodes 2,..6 cannot fit into the DFVS; so 0,1 have to be deleted
        for ub in 2..4 {
            let mut graph = graph.clone();
            let mut fvs = Vec::new();
            let applied = apply_rules_5_and_6(&mut graph, ub, &mut fvs);
            assert_eq!(applied, Some(true));
            fvs.sort_unstable();
            assert_eq!(fvs, vec![0, 1]);
            assert_eq!(graph.number_of_edges(), 0);
        }

        // ub is too low to fit nodes 0, 1 into fvs
        {
            let mut fvs = Vec::new();
            let mut graph = graph.clone();
            let applied = apply_rules_5_and_6(&mut graph, 1, &mut fvs);
            assert_eq!(applied, None);
        }
    }

    #[test]
    fn stress_rule_5() {
        stress_test_kernel(|graph, fvs, _| Some(apply_rule_5(graph, fvs)));
    }

    #[test]
    #[should_panic]
    fn stress_rules_5_and_6_too_low_ub() {
        stress_test_kernel(|graph, fvs, opt| {
            apply_rules_5_and_6(graph, opt.saturating_sub(1), fvs)
        });
    }

    #[test]
    fn stress_rules_5_and_6() {
        stress_test_kernel(|graph, fvs, opt| apply_rules_5_and_6(graph, opt, fvs));
    }

    #[test]
    fn stress_rules_5_and_6_higher_ub() {
        stress_test_kernel(|graph, fvs, opt| apply_rules_5_and_6(graph, opt + 1, fvs));
    }

    #[test]
    #[should_panic]
    fn stress_rule_6_too_low_ub() {
        stress_test_kernel(|graph, fvs, opt| apply_rule_6(graph, opt.saturating_sub(1), fvs));
    }

    #[test]
    fn stress_rule_6() {
        stress_test_kernel(|graph, fvs, opt| apply_rule_6(graph, opt, fvs));
    }

    #[test]
    fn stress_rule_6_higher_ub() {
        stress_test_kernel(|graph, fvs, opt| apply_rule_6(graph, opt + 1, fvs));
    }
}
