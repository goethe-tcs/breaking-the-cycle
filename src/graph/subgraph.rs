use super::*;
use crate::bitset::BitSet;
use std::collections::HashMap;

pub trait NodeMapperWriter {
    fn with_capacity(n: Node) -> Self;
    fn map_node_to(&mut self, old: Node, new: Node);
}

pub trait NodeMapperGetter {
    fn get(&self, old: Node) -> Option<Node>;
}

/// This Node Mapper cannot be read from, and all insertions are dumped.
/// It can be used to optimize a way the cost of producing a mapping if it is not used
pub struct WriteOnlyNodeMapper {}
impl NodeMapperWriter for WriteOnlyNodeMapper {
    fn with_capacity(_: Node) -> Self {
        Self {}
    }
    fn map_node_to(&mut self, _old: Node, _new: Node) {}
}

pub struct NodeMapper {
    mapping: HashMap<Node, Node>,
}

impl NodeMapperWriter for NodeMapper {
    fn with_capacity(n: Node) -> Self {
        Self {
            mapping: HashMap::with_capacity(n as usize),
        }
    }
    fn map_node_to(&mut self, old: Node, new: Node) {
        self.mapping.insert(old, new);
    }
}

impl NodeMapperGetter for NodeMapper {
    fn get(&self, old: Node) -> Option<Node> {
        Some(*self.mapping.get(&old)?)
    }
}

pub trait Concat {
    /// Takes a list of graphs and outputs a single graph containing disjoint copies of them all
    /// Let n_1, ..., n_k be the number of nodes in the input graph. Then node i of graph G_j becomes
    /// sum(n_1 + .. + n_{j-1}) + i
    fn concat<'a, IG: 'a + AdjacencyList, T: IntoIterator<Item = &'a IG> + Clone>(
        graphs: T,
    ) -> Self;
}

impl<G: GraphNew + GraphEdgeEditing> Concat for G {
    fn concat<'a, IG: 'a + AdjacencyList, T: IntoIterator<Item = &'a IG> + Clone>(
        graphs: T,
    ) -> Self {
        let total_n = graphs
            .clone()
            .into_iter()
            .map(|g| g.number_of_nodes() as usize)
            .sum();
        let mut result = G::new(total_n);

        let mut first_node: Node = 0;
        for g in graphs {
            // reserve memory
            for u in g.vertices() {
                for v in g.out_neighbors(u) {
                    result.add_edge(first_node + u, first_node + v);
                }
            }

            first_node += g.number_of_nodes();
        }

        result
    }
}

pub trait InducedSubgraph: Sized {
    /// Returns a new graph instance containing all nodes i with vertices\[i\] == true
    fn vertex_induced_as<M, Gout>(&self, vertices: &BitSet) -> (Gout, M)
    where
        M: NodeMapperGetter + NodeMapperWriter,
        Gout: GraphNew + GraphEdgeEditing;

    fn vertex_induced(&self, vertices: &BitSet) -> (Self, NodeMapper)
    where
        Self: GraphEdgeEditing,
    {
        self.vertex_induced_as(vertices)
    }
}

impl<G: GraphNew + GraphEdgeEditing + AdjacencyList + Sized> InducedSubgraph for G {
    fn vertex_induced_as<M, Gout>(&self, vertices: &BitSet) -> (Gout, M)
    where
        M: NodeMapperGetter + NodeMapperWriter,
        Gout: GraphNew + GraphEdgeEditing,
    {
        assert_eq!(vertices.len(), self.len());
        let new_n = vertices.cardinality();
        let mut result = Gout::new(new_n);

        // compute new node ids
        let mut mapping = M::with_capacity(new_n as Node);
        for (new, old) in vertices.iter().enumerate() {
            mapping.map_node_to(old as Node, new as Node);
        }

        for u in self.vertices() {
            if let Some(new_u) = mapping.get(u) {
                for new_v in self.out_neighbors(u).filter_map(|v| mapping.get(v)) {
                    result.add_edge(new_u, new_v);
                }
            }
        }

        (result, mapping)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_concat() {
        let g1 = AdjListMatrix::from(&[(0, 1), (2, 3)]);
        assert_eq!(g1.number_of_nodes(), 4);
        let g2 = AdjListMatrix::from(&[(0, 2)]);
        assert_eq!(g2.number_of_nodes(), 3);
        let g_empty = AdjListMatrix::new(0);
        let cc = AdjListMatrix::concat([&g1, &g_empty, &g1, &g2]);
        assert_eq!(cc.number_of_nodes(), 11);
        assert_eq!(cc.number_of_edges(), 5);
        assert_eq!(cc.edges(), vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 10)]);
    }

    #[test]
    fn test_induced() {
        let mut g = AdjListMatrix::new(6);
        for i in 0u32..4 {
            for j in 0u32..4 {
                g.add_edge(i, j);
            }
        }
        g.add_edge(4, 5);

        let bs = BitSet::from_slice(6, &[0, 1, 3, 5]);
        let (ind, mapping): (AdjListMatrix, NodeMapper) = g.vertex_induced(&bs);
        assert_eq!(ind.len(), bs.cardinality());
        assert!(g
            .vertices()
            .all(|u| { mapping.get(u).is_some() == bs[u as usize] }));

        let v_iso = mapping.get(5 as Node).unwrap();
        assert_eq!(ind.out_degree(v_iso), 0);
        assert_eq!(ind.in_degree(v_iso), 0);

        for u in [0, 1, 3].map(|u| mapping.get(u as Node).unwrap()) {
            assert_eq!(ind.in_degree(u), 3);
            assert_eq!(ind.out_degree(u), 3);
        }
    }
}
