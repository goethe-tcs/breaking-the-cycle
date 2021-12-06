use crate::graph::*;

pub trait ReductionState<G> {
    fn graph(&self) -> &G;
    fn graph_mut(&mut self) -> &mut G;
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

impl<G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest> PreprocessorReduction<G> {
    pub fn apply_rules_exhaustively(&mut self) {
        // TODO!
    }
}
