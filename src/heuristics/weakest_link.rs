use crate::graph::{AdjacencyListIn, GraphEdgeEditing, Node};
use std::cmp::min;

/// Linear time heuristic (in edges):
/// The goal is to find nodes with a minimal score,
/// where score(node) = min(indegree(node), outdegree(node)) in every round.
/// Out of all nodes with minimal score, we pick a min_node which has the highest
/// total degree and delete all incoming nodes of the min_node
/// (if indegree(node) < outdegree(node)) otherwise we delete all outgoing nodes
/// and put them into the dfvs.
/// Afterwards min_node is a sink or source
/// and we are applying reduction 3 exhaustively.
/// This algorithm needs reduction rule 1 (no self-loops).
pub fn weakest_link<G: GraphEdgeEditing + AdjacencyListIn + Clone, F: Fn() -> bool>(
    mut working_graph: G,
    stop_condition: F,
) -> Option<Vec<Node>> {
    let mut dfvs = Vec::new();
    let mut bucket_queue = BucketQueue::new_from_graph(&working_graph);
    while working_graph.number_of_edges() > 0 {
        if stop_condition() {
            return None;
        }

        // apply rule 3 exhaustively:
        while !bucket_queue.buckets[0].is_empty() {
            let node_zero = bucket_queue.buckets[0][0];
            // as long we still have sinks and sources:
            delete_node(&mut bucket_queue, node_zero, &mut working_graph);
        }
        // find optimal_node:
        let smallest_non_empty_bucket =
            bucket_queue.buckets.iter().find(|b| !b.is_empty()).unwrap();
        let optimal_node = *(smallest_non_empty_bucket
            .iter()
            .max_by_key(|&node| working_graph.total_degree(*node))
            .unwrap());

        // take out weakest_link:
        let neighbours_of_optimal_node: Vec<Node> =
            if working_graph.in_degree(optimal_node) < working_graph.out_degree(optimal_node) {
                working_graph.in_neighbors(optimal_node).collect()
            } else {
                working_graph.out_neighbors(optimal_node).collect()
            };
        delete_node(&mut bucket_queue, optimal_node, &mut working_graph);
        for node in neighbours_of_optimal_node {
            delete_node(&mut bucket_queue, node, &mut working_graph);
            dfvs.push(node)
        }
    }
    Some(dfvs)
}

pub fn weakest_link_no_stop_condition<G: GraphEdgeEditing + AdjacencyListIn + Clone>(
    working_graph: G,
) -> Vec<Node> {
    weakest_link(working_graph, || false).unwrap()
}

fn delete_node<G: GraphEdgeEditing + AdjacencyListIn + Clone>(
    bucket_queue: &mut BucketQueue,
    node: Node,
    working_graph: &mut G,
) {
    // delete node:
    bucket_queue.delete_node(node);
    working_graph.remove_edges_at_node(node);

    // update bucket queue for neighbours of node:
    let neighbours = working_graph
        .in_neighbors(node)
        .chain(working_graph.out_neighbors(node));
    for neighbour in neighbours {
        bucket_queue.update_node(neighbour, score(neighbour, working_graph));
    }
}

fn score<G: AdjacencyListIn>(node: Node, graph: &G) -> usize {
    min(graph.in_degree(node), graph.out_degree(node)) as usize
}

struct BucketQueue {
    buckets: Vec<Vec<Node>>,
    indices: Vec<(usize, usize)>, // lookup-table: node -> (bucket_id, index_in_bucket)
}

impl BucketQueue {
    fn new_from_graph<G: GraphEdgeEditing + AdjacencyListIn + Clone>(graph: &G) -> Self {
        // init BucketQueue:
        let mut buckets = vec![Vec::new(); graph.len()];
        let mut indices = Vec::with_capacity(graph.len());
        for node in 0..graph.number_of_nodes() {
            let bucked_id = score(node, graph);
            let index_in_bucket = buckets[bucked_id].len();
            buckets[bucked_id].push(node);
            indices.push((bucked_id, index_in_bucket))
        }
        Self { buckets, indices }
    }
    fn update_node(&mut self, node: Node, new_priority: usize) {
        self.delete_node(node); // delete node out of prior bucket
        self.indices[node as usize] = (new_priority, self.buckets[new_priority].len()); // set indices
        self.buckets[new_priority].push(node); // insert node into new bucket
    }
    fn delete_node(&mut self, node: Node) {
        let (bucket_id, index_in_bucket) = self.indices[node as usize];
        let last_node = *self.buckets[bucket_id].last().unwrap();
        self.buckets[bucket_id].swap_remove(index_in_bucket); // remove node out of bucket
        if last_node != node {
            // update index of switched last node:
            self.indices[last_node as usize].1 = index_in_bucket;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::graph::adj_array::AdjArrayIn;
    use crate::heuristics::weakest_link::weakest_link_no_stop_condition;

    #[test]
    fn test_0() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (2, 1)]);
        let dfvs = weakest_link_no_stop_condition(graph);
        assert_eq!(Vec::from([1, 3]), dfvs);
    }
    #[test]
    fn test_1() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 0), (0, 2), (0, 3), (1, 2), (2, 3), (3, 0)]);
        let dfvs = weakest_link_no_stop_condition(graph);
        assert_eq!(Vec::from([0]), dfvs);
    }
    #[test]
    fn test_2() {
        let graph = AdjArrayIn::from(&[
            (0, 1),
            (1, 0),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 4),
            (5, 6),
            (6, 7),
        ]);
        let dfvs = weakest_link_no_stop_condition(graph);
        assert_eq!(Vec::from([4, 0]), dfvs);
    }
}
