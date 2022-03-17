pub mod node_index_set;
pub mod set_vec;

use crate::bitset::BitSet;
use crate::graph::{AdjacencyListIn, GraphEdgeEditing, GraphOrder, InducedSubgraph, Node};

/// Creates a subgraph of the passed in graph by removing all nodes of the fvs
pub fn apply_fvs_to_graph<G, I>(graph: &G, fvs: I) -> G
where
    G: AdjacencyListIn + GraphOrder + GraphEdgeEditing,
    I: IntoIterator<Item = Node>,
{
    let bit_set = BitSet::new_all_set_but(graph.len(), fvs);

    let (graph, _node_mapping) = (*graph).vertex_induced(&bit_set);
    graph
}
