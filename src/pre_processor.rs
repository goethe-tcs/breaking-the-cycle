use crate::graph::Graph;

pub struct PreProcessor {
    graph: Graph,
}

pub struct PreProcessedInstance {
    og_graph: Graph,
    induced_graphs: Vec<Graph>,
    graph_labels: Vec<Vec<u32>>,
}

impl PreProcessedInstance {
    pub fn og_graph(&self) -> &Graph {
        &self.og_graph
    }
    pub fn induced_graphs(&self) -> &[Graph] {
        &self.induced_graphs
    }
    pub fn graph_labels(&self) -> &[Vec<u32>] {
        &self.graph_labels
    }
}

impl PreProcessor {
    // Assumes the graph contains no self-loops
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub fn reduce(self) -> PreProcessedInstance {
        let mut induced_graphs = vec![];
        let mut graph_labels = vec![];
        let sccs = self.graph.strongly_connected_components();

        // scc with cardinality 1 is irrelevant, as they are isolated vertices without self-loops
        for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
            let mut scc_g = Graph::new(scc.len());
            for (i, u) in scc.iter().enumerate() {
                for (j, v) in scc.iter().enumerate() {
                    if self.graph.has_edge(*u, *v) {
                        scc_g.add_edge(i as u32, j as u32);
                    }
                }
            }
            induced_graphs.push(scc_g);
            graph_labels.push(scc);
        }

        PreProcessedInstance {
            og_graph: self.graph,
            induced_graphs,
            graph_labels,
        }
    }
}
