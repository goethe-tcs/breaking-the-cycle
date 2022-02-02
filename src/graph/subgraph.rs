use super::*;
use crate::bitset::BitSet;

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
        M: node_mapper::Getter + node_mapper::Setter,
        Gout: GraphNew + GraphEdgeEditing;

    fn vertex_induced(&self, vertices: &BitSet) -> (Self, NodeMapper)
    where
        Self: GraphEdgeEditing,
    {
        self.vertex_induced_as(vertices)
    }

    /// Creates a subgraph where all nodes without edges are removed
    fn remove_disconnected_verts(&self) -> (Self, NodeMapper)
    where
        Self: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyListIn,
    {
        self.vertex_induced(&BitSet::new_all_unset_but(
            self.len(),
            self.vertices().filter(|&u| self.total_degree(u) > 0),
        ))
    }
}

impl<G: GraphNew + GraphEdgeEditing + AdjacencyList + Sized> InducedSubgraph for G {
    fn vertex_induced_as<M, Gout>(&self, vertices: &BitSet) -> (Gout, M)
    where
        M: node_mapper::Getter + node_mapper::Setter,
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
            if let Some(new_u) = mapping.new_id_of(u) {
                for new_v in self.out_neighbors(u).filter_map(|v| mapping.new_id_of(v)) {
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
    use crate::graph::adj_list_matrix::AdjListMatrixIn;

    #[test]
    fn test_concat() {
        let g1 = AdjListMatrixIn::from(&[(0, 1), (2, 3)]);
        assert_eq!(g1.number_of_nodes(), 4);
        let g2 = AdjListMatrixIn::from(&[(0, 2)]);
        assert_eq!(g2.number_of_nodes(), 3);
        let g_empty = AdjListMatrixIn::new(0);
        let cc = AdjListMatrixIn::concat([&g1, &g_empty, &g1, &g2]);
        assert_eq!(cc.number_of_nodes(), 11);
        assert_eq!(cc.number_of_edges(), 5);
        assert_eq!(cc.edges(), vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 10)]);
    }

    #[test]
    fn test_induced() {
        let mut g = AdjListMatrixIn::new(6);
        for i in 0u32..4 {
            for j in 0u32..4 {
                g.add_edge(i, j);
            }
        }
        g.add_edge(4, 5);

        let bs = BitSet::from_slice(6usize, &[0u32, 1u32, 3u32, 5u32]);
        let (ind, mapping): (AdjListMatrixIn, NodeMapper) = g.vertex_induced(&bs);
        assert_eq!(ind.len(), bs.cardinality());
        assert!(g
            .vertices()
            .all(|u| { mapping.new_id_of(u).is_some() == bs[u as usize] }));

        let v_iso = mapping.new_id_of(5 as Node).unwrap();
        assert_eq!(ind.out_degree(v_iso), 0);
        assert_eq!(ind.in_degree(v_iso), 0);

        for u in [0, 1, 3].map(|u| mapping.new_id_of(u as Node).unwrap()) {
            assert_eq!(ind.in_degree(u), 3);
            assert_eq!(ind.out_degree(u), 3);
        }
    }
}
