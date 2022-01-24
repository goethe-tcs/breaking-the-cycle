use crate::graph::*;

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
    pub fn apply_rules_exhaustively(&mut self) {
        apply_rules_exhaustively(&mut self.graph, &mut self.in_fvs)
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
}

/// applies rule 1-4 exhaustively
pub fn apply_rules_exhaustively<
    G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
) {
    apply_rule_1(graph, fvs);
    apply_rule_3(graph);
    loop {
        let rule_4 = apply_rule_4(graph, fvs);
        let rule_3 = apply_rule_3(graph);
        if !rule_4 && !rule_3 {
            break;
        }
    }
}

// TODO: implement more rules

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

        assert_eq!(test_pre_process.graph.edges().len(), 4);
        assert_eq!(test_pre_process.in_fvs.len(), 2);
    }

    #[test]
    fn test_rule_3() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_3();
        assert_eq!(test_pre_process.graph.edges().len(), 8);
        assert_eq!(test_pre_process.graph.out_degree(5), 0);
    }

    #[test]
    fn test_rule_4_neighbor_is_neighbor() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 3);
        assert_eq!(test_pre_process.graph.edges().len(), 0);
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
        assert_eq!(test_pre_process.graph.edges().len(), 10);
    }
    #[test]
    fn test_use_rules_exhaustively() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rules_exhaustively();
        assert!(!test_pre_process.apply_rule_1());
        assert!(!test_pre_process.apply_rule_3());
        assert!(!test_pre_process.apply_rule_4());
    }
}
