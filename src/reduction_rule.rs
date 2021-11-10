use crate::graph::Graph;
use bimap::BiMap;
use std::iter::FromIterator;

trait ReductionRule: From<Graph> {
    fn reduce(self) -> Box<dyn AppliedRule>;
}

trait AppliedRule {
    fn old_graph(&self) -> &Graph;
    fn new_graph(&self) -> &Graph;
    fn vertex_mapping(&self) -> &BiMap<u32, u32>;
}

pub struct SelfLoops {
    graph: Graph,
}

pub struct SelfLoopsResult {
    old_graph: Graph,
    new_graph: Graph,
    vertex_mapping: BiMap<u32, u32>,
    removed: Vec<u32>,
}

impl SelfLoopsResult {
    pub fn removed(&self) -> &[u32] {
        &self.removed
    }
}

impl AppliedRule for SelfLoopsResult {
    fn old_graph(&self) -> &Graph {
        &self.old_graph
    }

    fn new_graph(&self) -> &Graph {
        &self.new_graph
    }

    fn vertex_mapping(&self) -> &BiMap<u32, u32> {
        &self.vertex_mapping
    }
}

impl From<Graph> for SelfLoops {
    fn from(graph: Graph) -> Self {
        Self {
            graph,
        }
    }
}

impl ReductionRule for SelfLoops {
    fn reduce(self) -> Box<dyn AppliedRule> {
        let old_graph = self.graph;
        let mut new_graph = old_graph.clone();
        let removed: Vec<_> = old_graph.vertices().filter(|v| old_graph.has_edge(*v, *v)).collect();
        for u in &removed {
            new_graph.remove_edge(*u, *u);
        }
        let vertex_mapping = BiMap::from_iter(old_graph.vertices().map(|u| (u, u)));
        Box::new(SelfLoopsResult {
            old_graph,
            new_graph,
            vertex_mapping,
            removed
        })
    }
}