use super::*;
use crate::bitset::BitSet;
use itertools::Itertools;
use std::time::{Duration, Instant};

/// Implementation inspired by Abu-khzam et al. "Kernelization Algorithms for the Vertex Cover
/// Problem: Theory and Experiments."
pub fn apply_rule_crown<G: ReducibleGraph>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
    timeout: Option<Duration>,
) -> bool {
    use rand::prelude::SliceRandom;
    let mut rng = rand::thread_rng();
    let start_time = Instant::now();

    repeat_while(|| {
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
            graph
                .vertices()
                .filter(|&u| graph.undir_degree(u) == 0 || graph.has_dir_edges(u)),
        );

        if ignored_vertices.cardinality() == graph.len() {
            return false;
        }

        // get all undirected edges between not-ignored edges only from u -> v for u < v
        let mut edges = graph
            .vertices()
            .filter(|&u| !ignored_vertices[u as usize])
            .flat_map(|u| {
                graph
                    .undir_neighbors(u)
                    .map(move |v| (u, v))
                    .filter(|&(u, v)| u < v && !ignored_vertices[v as usize])
            })
            .collect_vec();

        'outer: for _attempt in 0..5 {
            if timeout
                .as_ref()
                .map_or(false, |t| start_time.elapsed() > *t)
            {
                return false;
            }

            // compute maximal matching and return all unmatched and un-ignored nodes
            let candidates = {
                edges.shuffle(&mut rng);

                let mut matched = ignored_vertices.clone();

                for &(u, v) in &edges {
                    if !matched[u as usize] && !matched[v as usize] {
                        matched.set_bit(u as usize);
                        matched.set_bit(v as usize);
                    }
                }

                if matched.cardinality() + 1 >= graph.len() {
                    continue 'outer;
                }

                matched.not();
                matched
            };

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

            return true;
        }

        false
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn stress_crown() {
        stress_test_kernel(|graph, fvs, _| Some(apply_rule_crown(graph, fvs, None)));
    }
}
