use crate::graph::*;
use bimap::BiMap;

trait ReductionRule<G>: From<G> {
    fn reduce(self) -> Box<dyn AppliedRule<G>>;
}

trait AppliedRule<G> {
    fn graph(&self) -> &G;
    fn removed_edges(&self) -> Option<&[Edge]>;
    fn removed_vertices(&self) -> Option<&[Node]>;
    fn vertex_mapping(&self) -> Option<&BiMap<Node, Node>>;
    fn added_edges(&self) -> Option<&[Edge]>;
}

pub struct SelfLoops<G> {
    graph: G,
}

pub struct SelfLoopsResult<G> {
    graph: G,
    removed_edges: Vec<Edge>,
}

impl<G> AppliedRule<G> for SelfLoopsResult<G> {
    fn graph(&self) -> &G {
        &self.graph
    }

    fn removed_edges(&self) -> Option<&[Edge]> {
        Some(&self.removed_edges)
    }

    fn removed_vertices(&self) -> Option<&[Node]> {
        None
    }

    fn vertex_mapping(&self) -> Option<&BiMap<Node, Node>> {
        None
    }

    fn added_edges(&self) -> Option<&[Edge]> {
        None
    }
}

impl<G> From<G> for SelfLoops<G> {
    fn from(graph: G) -> Self {
        Self { graph }
    }
}

impl<G: 'static +  AdjecencyList + AdjecencyTest + GraphEdgeEditing + GraphNew + Clone> ReductionRule<G> for SelfLoops<G> {
    fn reduce(self) -> Box<dyn AppliedRule<G>> {
        let old_graph = self.graph;
        let mut new_graph = old_graph.clone();
        let removed_edges: Vec<_> = old_graph
            .vertices()
            .filter(|v| old_graph.has_edge(*v, *v))
            .map(|v| (v, v))
            .collect();
        for (u, v) in &removed_edges {
            new_graph.remove_edge(*u, *v);
        }
        Box::new(SelfLoopsResult {
            graph: new_graph,
            removed_edges,
        })
    }
}
