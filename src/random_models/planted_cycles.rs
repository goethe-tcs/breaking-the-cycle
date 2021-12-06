use crate::graph::{generators::*, *};
use rand::{prelude::IteratorRandom, Rng};

/// Generates a random graph over n nodes and ~n*avg_deg/2 edges,
/// with a num_cycles directed cycles and a DFVS of size k. The expected size of a cycle
/// if avgCircleLen. Make sure that n is chosen sufficently larger
pub fn generate_planted_cycles<G, R>(
    rng: &mut R,
    n: Node,
    avg_deg: f64,
    k: Node,
    num_cycles: Node,
    cycle_len: Node,
) -> (G, Vec<Node>)
where
    G: GraphNew
        + GraphEdgeEditing
        + subgraph::Concat
        + AdjacencyList
        + AdjacencyTest
        + Traversal
        + Sized,
    R: Rng,
{
    assert!(avg_deg < n as f64 / 2.0);
    assert!(n > num_cycles * cycle_len + k);
    assert!(num_cycles >= k);

    let nodes_in_cycles = num_cycles * cycle_len;
    let nodes_in_mst = n - nodes_in_cycles;

    let (mst, mst_root): (G, Node) = random_mst(rng, nodes_in_mst as usize);

    let mut ranks: Vec<f64> = mst
        .bfs(mst_root)
        .ranking()
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect();
    ranks.resize(n as usize, 0.0);

    let mut result_g = G::concat([&mst, &G::new((n - nodes_in_mst) as usize)]);

    // draw k pair-wise disjoint nodes from the mst
    let fvs: Vec<Node> = mst.vertices().into_iter().choose_multiple(rng, k as usize);

    // introduce numCircle cycles, each of length cycleLen,
    for ci in 0..num_cycles {
        // the first k cycles are placed at the center in order; the remaining are placed at random centers
        let center_node = if ci < k {
            fvs[ci as usize]
        } else {
            *fvs.iter().choose(rng).unwrap()
        };

        let first_node = nodes_in_mst + cycle_len * ci;

        // construct a path and then make it a cycle
        result_g.add_edge(center_node, first_node);
        result_g.connect_path((first_node..first_node + cycle_len).collect::<Vec<Node>>());
        result_g.add_edge(first_node + cycle_len - 1, center_node);

        // update the ranks of the cycle's nodes
        for i in 0..cycle_len {
            ranks[(first_node + i) as usize] = ranks[center_node as usize]
                + ((i + ci * cycle_len + 1) as f64) / ((nodes_in_cycles + 2) as f64);
        }
    }
    let vertices: Vec<Node> = result_g.vertices().collect();

    // introduce additional edges that do not close additional cycles as they only
    // connect nodes of lower ranks to nodes of higher ranks
    let target_m = (n as f64 * avg_deg / 2.0) as usize;
    while result_g.number_of_edges() < target_m {
        let u = vertices[rng.gen_range(0..vertices.len())];
        let v = vertices[rng.gen_range(0..vertices.len())];

        if u == v {
            continue;
        }
        assert_ne!(ranks[u as usize], ranks[v as usize]);

        if ranks[u as usize] < ranks[v as usize] {
            if result_g.has_edge(u, v) {
                continue;
            }
            result_g.add_edge(u, v);
        } else {
            if result_g.has_edge(v, u) {
                continue;
            }
            result_g.add_edge(v, u);
        }
    }

    (result_g, fvs)
}

#[test]
fn test_random_graph() {
    use crate::bitset::BitSet;

    for n in [10, 11, 12, 20, 30, 50] {
        for avg_deg in [1.0, 2.0, 4.0] {
            for k in [1, 2, 5] {
                for num_cycles in [k, k + 1, 2 * k] {
                    if k == 0 && num_cycles > 0 {
                        // we cannot have cycles with out seed nodes
                        continue;
                    }

                    for cycle_len in [1, 2, 5] {
                        if n <= num_cycles * cycle_len + k {
                            continue;
                        }

                        let res = generate_planted_cycles(
                            &mut rand::thread_rng(),
                            n,
                            avg_deg,
                            k,
                            num_cycles,
                            cycle_len,
                        );
                        let graph: AdjListMatrix = res.0;
                        let fvs = res.1;

                        assert_eq!(graph.number_of_nodes(), n);
                        assert_eq!(fvs.len(), k as usize);
                        assert!(graph.number_of_edges() + 1 >= (n as f64 * avg_deg / 2.0) as usize);

                        assert_eq!(graph.is_acyclic(), k == 0, "\n k={}, \n {:?}", k, graph);

                        let not_in_fvs = {
                            // todo: use new_all_set_but
                            let mut set = BitSet::new_all_set(graph.len());
                            for v in fvs {
                                set.unset_bit(v as usize);
                            }
                            set
                        };

                        let remaining = graph.vertex_induced(&not_in_fvs).0;

                        assert!(remaining.is_acyclic());
                    }
                }
            }
        }
    }
}
