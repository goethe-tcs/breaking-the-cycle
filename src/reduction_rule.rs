use crate::graph::Graph;
use bimap::BiMap;

trait ReductionRule: From<Graph> {
    fn reduce(self) -> Box<dyn AppliedRule>;
}

trait AppliedRule {
    fn graph(&self) -> &Graph;
    fn removed_edges(&self) -> Option<&[(u32, u32)]>;
    fn removed_vertices(&self) -> Option<&[u32]>;
    fn vertex_mapping(&self) -> Option<&BiMap<u32, u32>>;
    fn added_edges(&self) -> Option<&[(u32, u32)]>;
}

pub struct SelfLoops {
    graph: Graph,
}

pub struct SelfLoopsResult {
    graph: Graph,
    removed_edges: Vec<(u32, u32)>,
}

impl AppliedRule for SelfLoopsResult {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn removed_edges(&self) -> Option<&[(u32, u32)]> {
        Some(&self.removed_edges)
    }

    fn removed_vertices(&self) -> Option<&[u32]> {
        None
    }

    fn vertex_mapping(&self) -> Option<&BiMap<u32, u32>> {
        None
    }

    fn added_edges(&self) -> Option<&[(u32, u32)]> {
        None
    }
}

impl From<Graph> for SelfLoops {
    fn from(graph: Graph) -> Self {
        Self { graph }
    }
}

impl ReductionRule for SelfLoops {
    fn reduce(self) -> Box<dyn AppliedRule> {
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
