use super::*;
use crate::bitset::BitSet;
use itertools::Itertools;

/// Implementation inspired by Abu-khzam et al. "Kernelization Algorithms for the Vertex Cover
/// Problem: Theory and Experiments."
pub fn apply_rule_crown<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    use rand::prelude::SliceRandom;
    let mut rng = rand::thread_rng();

    loop {
        let nodes_matched_in = |matching: &[(Node, Node)]| {
            let mut matched = BitSet::new(graph.len());
            for &(u, v) in matching {
                matched.set_bit(u as usize);
                matched.set_bit(v as usize);
            }
            matched
        };

        let open_neighbors_of = |nodes: &BitSet| {
            let mut neighbors = BitSet::new_all_unset_but(
                graph.len(),
                nodes
                    .iter()
                    .flat_map(|c| graph.undir_neighbors(c as Node).map(|v| v as usize)),
            );
            neighbors.and_not(nodes);
            neighbors
        };

        let ignored_vertices = BitSet::new_all_unset_but(
            graph.len(),
            graph.vertices().filter(|&u| {
                //graph.total_degree(u) == 0 || graph.total_degree(u) != 2 * graph.undir_degree(u)
                graph.undir_degree(u) == 0
                    || (graph.in_degree(u) > graph.undir_degree(u)
                        && graph.out_degree(u) > graph.undir_degree(u))
            }),
        );

        if ignored_vertices.cardinality() == graph.len() {
            return applied;
        }

        let mut edges = graph
            .edges_iter()
            .filter(|&(u, v)| !ignored_vertices[u as usize] && !ignored_vertices[v as usize])
            .collect_vec();

        if edges.is_empty() {
            return applied;
        }

        let mut applied_local = false;
        for _ in 0..5 {
            edges.shuffle(&mut rng);

            let mut matched = ignored_vertices.clone();

            for &(u, v) in &edges {
                if !matched[u as usize] && !matched[v as usize] {
                    matched.set_bit(u as usize);
                    matched.set_bit(v as usize);
                }
            }

            if matched.cardinality() == graph.len() {
                continue;
            }

            let mut candidates = matched;
            candidates.not();

            if candidates.cardinality() <= 1 {
                continue;
            }

            // assert candidates is an independent set
            debug_assert!(!candidates.iter().any(|u| candidates
                .iter()
                .any(|v| graph.has_edge(u as Node, v as Node))));

            let neighbors = open_neighbors_of(&candidates);

            let maximum_matching = graph.maximum_bipartite_matching(&candidates, &neighbors);

            let head = if maximum_matching.len() == neighbors.cardinality() {
                neighbors
            } else if maximum_matching.len() == candidates.cardinality() {
                continue;
            } else {
                let mut crown = candidates;
                crown.and_not(&nodes_matched_in(&maximum_matching));

                loop {
                    let head = open_neighbors_of(&crown);

                    let mut new_crown = crown.clone();

                    maximum_matching
                        .iter()
                        .flat_map(|&(c, h)| {
                            if head[h as usize] {
                                Some(c as usize)
                            } else {
                                None
                            }
                        })
                        .for_each(|c| {
                            new_crown.set_bit(c);
                        });

                    if crown.cardinality() == new_crown.cardinality() {
                        // assert crown is an independent set
                        debug_assert!(!crown
                            .iter()
                            .any(|u| crown.iter().any(|v| graph.has_edge(u as Node, v as Node))));

                        // assert no edge between crown and body (i.e. all edges out of crown go into head)
                        debug_assert!(crown
                            .iter()
                            .all(|u| graph.undir_neighbors(u as Node).all(|v| head[v as usize])));

                        break head;
                    }

                    crown = new_crown;
                }
            };

            if head.cardinality() == 0 {
                continue;
            }

            let mut to_delete = head.iter().map(|u| u as Node).collect_vec();

            graph.remove_edges_of_nodes(&to_delete);
            fvs.append(&mut to_delete);

            applied_local = true;
            break;
        }

        if !applied_local {
            return applied;
        }

        applied = true;
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn stress_crown() {
        stress_test_kernel(|graph, fvs, _| Some(apply_rule_crown(graph, fvs)));
    }
}
