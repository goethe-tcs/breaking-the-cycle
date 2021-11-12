use crate::graph::*;

pub struct PreProcessor<G: AdjecencyList + AdjecencyTest + GraphManipulation + GraphNew> {
    graph: G,
}

pub struct PreProcessedInstance<G: AdjecencyList + AdjecencyTest + GraphManipulation + GraphNew> {
    og_graph: G,
    induced_graphs: Vec<G>,
    graph_labels: Vec<Vec<u32>>,
}

impl<G: AdjecencyList + AdjecencyTest + GraphManipulation + GraphNew> PreProcessedInstance<G> {
    pub fn og_graph(&self) -> &G {
        &self.og_graph
    }
    pub fn induced_graphs(&self) -> &[G] {
        &self.induced_graphs
    }
    pub fn graph_labels(&self) -> &[Vec<u32>] {
        &self.graph_labels
    }
}

impl<G: AdjecencyList + AdjecencyTest + GraphManipulation + GraphNew> PreProcessor<G> {
    // Assumes the graph contains no self-loops
    pub fn new(graph: G) -> Self {
        Self { graph }
    }

    pub fn reduce(self) -> PreProcessedInstance<G> {
        let mut induced_graphs = vec![];
        let mut graph_labels = vec![];
        let sccs = self.graph.strongly_connected_components();

        // scc with cardinality 1 is irrelevant, as they are isolated vertices without self-loops
        for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
            let mut scc_g = G::new(scc.len());
            for (i, u) in scc.iter().enumerate() {
                for (j, v) in scc.iter().enumerate() {
                    if self.graph.has_edge(*u, *v) {
                        scc_g.add_edge(i as Node, j as Node);
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

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::graph::AdjListMatrix;

    #[test]
    fn pre_processor() {
        let mut graph = AdjListMatrix::new(8);
        let edges = [
            (0, 1),
            (1, 2),
            (1, 4),
            (1, 5),
            (2, 6),
            (2, 3),
            (3, 7),
            (3, 2),
            (4, 0),
            (4, 5),
            (5, 6),
            (6, 5),
            (7, 6),
            (7, 3),
        ];
        graph.add_edges(&edges);

        // edges between sccs are not contained in the union graph of each scc induced graph
        let edges_of_sccs: Vec<_> = edges
            .iter()
            .filter(|(u, v)| {
                (*u, *v) != (1, 2)
                    && (*u, *v) != (1, 5)
                    && (*u, *v) != (2, 6)
                    && (*u, *v) != (4, 5)
                    && (*u, *v) != (7, 6)
            })
            .copied()
            .collect();
        let mut sccs_graph = AdjListMatrix::new(8);
        sccs_graph.add_edges(&edges_of_sccs);

        let ppi = PreProcessor::new(graph).reduce();
        assert_eq!(ppi.induced_graphs.len(), 3);
        let order: Node = ppi.induced_graphs.iter().map(|g| g.number_of_nodes()).sum();
        assert_eq!(ppi.og_graph.number_of_nodes(), order);

        let mut union_graph = AdjListMatrix::new(ppi.og_graph.number_of_nodes() as usize);
        for (g, l) in ppi.induced_graphs.iter().zip(ppi.graph_labels.iter()) {
            for v in g.vertices() {
                for u in g.out_neighbors(v) {
                    union_graph.add_edge(l[v as usize], l[*u as usize]);
                }
            }
        }

        for u in union_graph.vertices() {
            for v in union_graph.out_neighbors(u) {
                assert!(sccs_graph.has_edge(u, *v));
            }
        }

        for u in sccs_graph.vertices() {
            for v in sccs_graph.out_neighbors(u) {
                assert!(union_graph.has_edge(u, *v));
            }
        }
    }
}
