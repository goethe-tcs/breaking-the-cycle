use crate::bitset::BitSet;
use crate::graph::network_flow::{EdmondsKarp, ResidualBitMatrix, ResidualNetwork};
use crate::graph::*;
use fxhash::FxHashSet;
use itertools::Itertools;
use std::iter::FromIterator;

pub trait ReductionState<G> {
    fn graph(&self) -> &G;
    fn graph_mut(&mut self) -> &mut G;

    fn fvs(&self) -> &[Node];
    fn add_to_fvs(&mut self, u: Node);
}

pub struct PreprocessorReduction<G> {
    graph: G,
    in_fvs: Vec<Node>,
}

impl<G> ReductionState<G> for PreprocessorReduction<G> {
    fn graph(&self) -> &G {
        &self.graph
    }
    fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    fn fvs(&self) -> &[Node] {
        &self.in_fvs
    }
    fn add_to_fvs(&mut self, u: Node) {
        self.in_fvs.push(u);
    }
}

impl<G> From<G> for PreprocessorReduction<G> {
    fn from(graph: G) -> Self {
        Self {
            graph,
            in_fvs: Vec::new(),
        }
    }
}

impl<G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn>
    PreprocessorReduction<G>
{
    // added AdjacencyListIn so we can use self.graph.out_degree()
    /// applies rule 1-4 exhaustively
    /// rule 5 is optional, because it runs slow
    pub fn apply_rules_exhaustively(&mut self, with_rule_5: bool) {
        apply_rules_exhaustively(&mut self.graph, &mut self.in_fvs, with_rule_5)
    }

    pub fn apply_rule_1(&mut self) -> bool {
        apply_rule_1(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_rule_3(&mut self) -> bool {
        apply_rule_3(&mut self.graph)
    }

    pub fn apply_rule_4(&mut self) -> bool {
        apply_rule_4(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_rule_5(&mut self) -> bool {
        apply_rule_5(&mut self.graph)
    }

    pub fn apply_di_cliques_reduction(&mut self) -> bool {
        apply_di_cliques_reduction(&mut self.graph, &mut self.in_fvs)
    }
}

/// applies rule 1-4 + 5 exhaustively
/// each reduction rule returns true, if it performed a reduction. false otherwise.
/// reduction rule 5 is much slower, than rule 1-4.
/// can take a minute if graph has about 10.000 nodes.
pub fn apply_rules_exhaustively<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
    with_expensive_rules: bool,
) {
    apply_rule_1(graph, fvs);
    loop {
        let rule_3 = apply_rule_3(graph);
        let rule_4 = apply_rule_4(graph, fvs);
        let rule_di_cliques = apply_di_cliques_reduction(graph, fvs);
        if !(rule_3 || rule_4 || rule_di_cliques) {
            break;
        }
    }
    if with_expensive_rules {
        apply_rule_5(graph);
    }
}

/// rule 1 - self-loop
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_1<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
) -> bool {
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
pub fn apply_rule_3<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
) -> bool {
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
pub fn apply_rule_4<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.in_degree(u) == 1 {
            let neighbor = graph.in_neighbors(u).next().unwrap();
            let out_neighbors: Vec<Node> = graph.out_neighbors(u).collect();
            if out_neighbors.contains(&neighbor) {
                fvs.push(neighbor);
                graph.remove_edges_at_node(neighbor);
            } else {
                for out_neighbor in out_neighbors {
                    graph.try_add_edge(neighbor, out_neighbor);
                }
            }
            graph.remove_edges_at_node(u);
            applied = true;
        } else if graph.out_degree(u) == 1 {
            let neighbor = graph.out_neighbors(u).next().unwrap();
            let in_neighbors: Vec<Node> = graph.in_neighbors(u).collect();
            if in_neighbors.contains(&neighbor) {
                fvs.push(neighbor);
                graph.remove_edges_at_node(neighbor);
            } else {
                for in_neighbor in in_neighbors {
                    graph.try_add_edge(in_neighbor, neighbor);
                }
            }
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }
    applied
}

/// Converts a graph into a Vector of BitSets. But every node v of the graph is split into two nodes.
/// One node has the ingoing edges of v and one has the outgoing edges of v.
/// Also adds an edge from the node with ingoing edges to the node with outgoing edges.
pub fn create_capacity_for_many_petals<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
) -> Vec<BitSet> {
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
fn perform_petal_reduction<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    node: Node,
    capacity: &mut [BitSet],
) {
    // removing edges from capacity
    capacity[(node + graph.len() as Node) as usize].unset_all();
    capacity[node as usize].unset_bit((node + (graph.len() as Node)) as usize);
    for bit_vector in capacity.iter_mut() {
        bit_vector.unset_bit(node as usize);
    }

    // must be collected before edges at node get removed.
    let in_neighbors: Vec<_> = graph.in_neighbors(node).collect();
    let out_neighbors: Vec<_> = graph.out_neighbors(node).collect();

    // removing edges from capacity graph
    graph.remove_edges_at_node(node);

    // add all possible edges for (in_neighbors, out_neighbors)
    for in_neighbor in in_neighbors {
        for &out_neighbor in &out_neighbors {
            let edge_from = (in_neighbor as usize) + graph.len();
            let edge_to = out_neighbor as usize;
            capacity[edge_from].set_bit(edge_to); // adds new edge to capacity

            graph.try_add_edge(in_neighbor, out_neighbor); // adds new edge to graph
        }
    }
}

/// Checks for each node u on a given graph, if u has exactly one petal.
/// If so, the graph gets reduced. That means, the node u with petal=1 gets deleted and all possible
/// Edges (in_neighbors(u), out_neighbour(u)) are added to the graph.
pub fn apply_rule_5<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
) -> bool {
    let mut applied = false;

    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        // prepare the capacity to run num_disjoint()
        capacity[node as usize].unset_bit(node as usize + graph.len());
        let petal_bit_matrix = ResidualBitMatrix::from_capacity_and_labels(
            capacity,
            labels,
            node + graph.len() as Node,
            node,
        );

        let mut ec = EdmondsKarp::new(petal_bit_matrix);
        ec.set_remember_changes(true);
        let petal_count = ec.count_num_disjoint_upto(2);
        ec.undo_changes();
        (capacity, labels) = ec.take(); // needed because capacity and labels is moved when petal_bit_matrix is created

        // actual reduction of graph (if petal_count = 1)
        if petal_count == 1 {
            applied = true;
            perform_petal_reduction(graph, node as Node, &mut capacity);
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }
    applied
}

/// Safe Di-Cliques Reduction, requires self loops to be deleted
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_di_cliques_reduction<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.out_degree(u) == 0 || graph.in_degree(u) == 0 {
            continue;
        }
        // creating an intersection of node u's neighborhood (nodes with 2 clique to u)
        let intersection = {
            if graph.out_degree(u) < graph.in_degree(u) {
                let set = FxHashSet::from_iter(graph.out_neighbors(u));
                graph
                    .in_neighbors(u)
                    .filter(|v| set.contains(v))
                    .collect_vec()
            } else {
                let set = FxHashSet::from_iter(graph.in_neighbors(u));
                graph
                    .out_neighbors(u)
                    .filter(|v| set.contains(v))
                    .collect_vec()
            }
        };
        // check whether u is in more than one clique. We can skip u if it is
        if intersection.is_empty()
            || intersection
                .iter()
                .cartesian_product(intersection.iter())
                .any(|(&x, &y)| x != y && !graph.has_edge(x, y))
        {
            continue;
        }
        // if u is only in 1 clique we check here if we can safely delete all nodes but u from that
        // clique. Either u has only out- or in-going edges or there is no circle back to u
        // outside of the clique. This last point is checked via breadth first search
        if graph.in_degree(u) as usize == intersection.len()
            || graph.out_degree(u) as usize == intersection.len()
            || !graph.is_node_on_cycle_after_deleting(u, intersection.clone())
        {
            for node in intersection {
                fvs.push(node);
                graph.remove_edges_at_node(node);
                applied = true;
            }
        }
    }
    applied
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;

    fn create_test_pre_processor() -> PreprocessorReduction<AdjArrayIn> {
        let graph = AdjArrayIn::from(&[
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

        let test_pre_process = PreprocessorReduction {
            graph,
            in_fvs: vec![],
        };

        test_pre_process
    }

    fn create_test_pre_processor_2() -> PreprocessorReduction<AdjArrayIn> {
        let graph = AdjArrayIn::from(&[
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

        let test_pre_process = PreprocessorReduction {
            graph,
            in_fvs: vec![],
        };

        test_pre_process
    }

    #[test]
    fn test_rule_1() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_1();

        assert_eq!(test_pre_process.graph.number_of_edges(), 4);
        assert_eq!(test_pre_process.in_fvs.len(), 2);
    }

    #[test]
    fn test_rule_3() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_3();
        assert_eq!(test_pre_process.graph.number_of_edges(), 8);
        assert_eq!(test_pre_process.graph.out_degree(5), 0);
    }

    #[test]
    fn test_rule_4_neighbor_is_neighbor() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 3);
        assert_eq!(test_pre_process.graph.number_of_edges(), 0);
    }

    #[test]
    fn test_rule_4_neighbor_not_neighbor() {
        let graph = AdjArrayIn::from(&[
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

        let mut test_pre_process = PreprocessorReduction {
            graph,
            in_fvs: vec![],
        };

        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 0);
        assert_eq!(test_pre_process.graph.number_of_edges(), 10);
    }

    #[test]
    fn test_use_rules_exhaustively() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rules_exhaustively(false);
        assert!(!test_pre_process.apply_rule_1());
        assert!(!test_pre_process.apply_rule_3());
        assert!(!test_pre_process.apply_rule_4());
    }

    fn create_circle(graph_size: usize) -> AdjArrayIn {
        let mut graph = AdjArrayIn::new(graph_size);
        for i in 0..graph_size - 1 {
            graph.add_edge(i as Node, (i + 1) as Node);
        }
        graph.add_edge((graph_size - 1) as Node, 0);
        graph
    }

    #[test]
    fn test_create_capacity_for_many_petals() {
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
    fn test_perform_petal_reduction() {
        let tested_graph_len = 4;
        let mut graph = create_circle(tested_graph_len);
        let mut capacity = create_capacity_for_many_petals(&mut graph);
        perform_petal_reduction(&mut graph, 0, &mut capacity);
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
    fn apply_rule_5() {
        let tested_graph_len = 4;
        let graph = create_circle(tested_graph_len);

        let mut test_pre_process = PreprocessorReduction {
            graph,
            in_fvs: vec![],
        };

        test_pre_process.apply_rule_5();

        let expected_edges_graph = vec![(3, 3)];

        assert_eq!(
            0,
            test_pre_process.graph.in_degree(0) + test_pre_process.graph.out_degree(0)
        );

        for from_node in 0..test_pre_process.graph.len() {
            for to_node in 0..test_pre_process.graph.len() {
                if expected_edges_graph.contains(&(from_node, to_node)) {
                    assert!(test_pre_process
                        .graph
                        .has_edge(from_node as Node, to_node as Node));
                } else {
                    assert!(!test_pre_process
                        .graph
                        .has_edge(from_node as Node, to_node as Node));
                }
            }
        }
    }
    #[test]
    fn test_di_cliques_reduction() {
        let graph = AdjArrayIn::from(&[
            (0, 1),
            (0, 3),
            (0, 6),
            (1, 3),
            (1, 4),
            (1, 6),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 6),
            (5, 2),
            (5, 4),
            (5, 6),
            (6, 0),
            (6, 3),
            (6, 4),
            (6, 7),
            (6, 8),
            (7, 6),
            (7, 8),
            (8, 6),
            (8, 7),
        ]);

        let mut test_pre_process = PreprocessorReduction {
            graph,
            in_fvs: vec![],
        };
        test_pre_process.apply_di_cliques_reduction();
        assert_eq!(test_pre_process.fvs(), vec![4, 5, 1, 6, 8]);
    }
}
