use crate::graph::{AdjacencyList, GraphEdgeEditing, Node, Traversal};
use crate::utils::int_iterator::IntegerIterators;
use crate::utils::int_subsets::AllIntSubsets;
use bitintr::Popcnt;
use itertools::Itertools;

/// A simple implementation of an exhaustive search to provide cross-validation
/// for more complicated algorithms.
///
/// The idea is enumerate all solutions with a given number of nodes in the DFVS. If no solution
/// can be obtained, we iteratively increase the number of nodes in the DFVS until we find a match.
/// This way the first feasible solution is in fact the smallest one.
pub fn exhaustive_search<G>(
    graph: &G,
    lower_bound: Option<Node>,
    upper_bound: Option<Node>,
) -> Vec<Node>
where
    G: Clone + AdjacencyList + GraphEdgeEditing,
{
    assert!(graph.number_of_nodes() < 64);

    let all_possible_solutions =
        AllIntSubsets::start_with_bits_set(lower_bound.unwrap_or(0), graph.len() as u32);

    let smallest_solution = all_possible_solutions
        .take_while(|&x| upper_bound.map_or(true, |up| x.popcnt() <= up.into()))
        .find(|&solution_mask| {
            let mut graph = graph.clone();
            for node_to_delete in solution_mask.iter_ones() {
                graph.remove_edges_at_node(node_to_delete);
            }

            graph.is_acyclic()
        })
        .unwrap();

    smallest_solution
        .iter_ones()
        .map(|u| u as Node)
        .collect_vec()
}

/// Executes `exhaustive_search` with bounds set to verify optimality of a known solution as fast as possible.
pub fn exhaustive_search_verify_optimality<G>(graph: &G, size_of_known_solution: Node) -> bool
where
    G: Clone + AdjacencyList + GraphEdgeEditing,
{
    let lower = if size_of_known_solution > 0 {
        size_of_known_solution - 1
    } else {
        0
    };
    exhaustive_search(graph, Some(lower), Some(size_of_known_solution + 1)).len()
        == size_of_known_solution as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;

    #[test]
    fn exhaustive_search_loops() {
        let graph = AdjArrayIn::from(&[(0, 0), (2, 2)]);
        assert_eq!(exhaustive_search(&graph, None, None), vec![0, 2]);
        assert_eq!(exhaustive_search(&graph, Some(1), None), vec![0, 2]);
        assert_eq!(exhaustive_search(&graph, Some(2), None), vec![0, 2]);
        assert_eq!(exhaustive_search(&graph, Some(3), None), vec![0, 1, 2]);
    }

    #[test]
    fn exhaustive_search_verify_opt() {
        assert!(exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            2
        ));

        assert!(!exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            3
        ));
    }
}
