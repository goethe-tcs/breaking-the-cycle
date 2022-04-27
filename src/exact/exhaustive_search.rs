use crate::graph::{AdjacencyList, GraphEdgeEditing, Node, Traversal};
use crate::utils::*;
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
) -> Option<Vec<Node>>
where
    G: Clone + AdjacencyList + GraphEdgeEditing,
{
    assert!(graph.number_of_nodes() < 64);

    let all_possible_solutions =
        AllIntSubsets::start_with_bits_set(lower_bound.unwrap_or(0), graph.len() as u32);

    let smallest_solution = all_possible_solutions
        .take_while(|&x| upper_bound.map_or(true, |up| x.count_ones() <= up))
        .find(|&sol| is_valid_dfvs(graph, sol.iter_ones()))?;

    Some(
        smallest_solution
            .iter_ones()
            .map(|u| u as Node)
            .collect_vec(),
    )
}

/// Executes `exhaustive_search` with bounds set to verify optimality of a known solution as fast as possible.
pub fn exhaustive_search_verify_optimality<G>(graph: &G, candidate: &[Node]) -> bool
where
    G: Clone + AdjacencyList + GraphEdgeEditing,
{
    let smaller_solution_exists = !candidate.is_empty()
        && exhaustive_search(
            graph,
            Some(candidate.len() as Node - 1),
            Some(candidate.len() as Node - 1),
        )
        .is_some();

    !smaller_solution_exists && is_valid_dfvs(graph, candidate.iter().copied())
}

fn is_valid_dfvs<G, I>(graph: &G, candidate: I) -> bool
where
    G: Clone + AdjacencyList + GraphEdgeEditing,
    I: IntoIterator<Item = Node>,
{
    let mut graph = graph.clone();
    for u in candidate {
        graph.remove_edges_at_node(u);
    }
    graph.is_acyclic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;

    #[test]
    fn exhaustive_search_loops() {
        let graph = AdjArrayIn::from(&[(0, 0), (2, 2)]);
        assert_eq!(exhaustive_search(&graph, None, None), Some(vec![0, 2]));
        assert_eq!(exhaustive_search(&graph, Some(1), None), Some(vec![0, 2]));
        assert_eq!(exhaustive_search(&graph, Some(2), None), Some(vec![0, 2]));
        assert_eq!(
            exhaustive_search(&graph, Some(3), None),
            Some(vec![0, 1, 2])
        );
    }

    #[test]
    fn exhaustive_search_verify_opt() {
        assert!(exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            &[0, 2]
        ));

        assert!(!exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            &[]
        ));

        assert!(!exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            &[0, 1]
        ));

        assert!(!exhaustive_search_verify_optimality(
            &AdjArrayIn::from(&[(0, 0), (2, 2)]),
            &[0, 1, 2]
        ));
    }
}
