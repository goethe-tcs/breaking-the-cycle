use super::super::utils::set_vec::HashSetVec;
use crate::graph::{AdjacencyListIn, GraphEdgeEditing, GraphOrder, Node};
use fxhash::{FxBuildHasher, FxHashSet};

/// Used to store progress during the local search
#[derive(Clone)]
pub struct TopologicalConfig<'a, G> {
    pub(crate) graph: &'a G,
    pub(crate) s: Vec<Node>,
    pub(crate) not_in_s: HashSetVec<Node>,
}

impl<'a, G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone> TopologicalConfig<'a, G> {
    /// Creates an empty TopologicalConfig
    pub fn new(graph: &'a G) -> Self {
        Self {
            not_in_s: HashSetVec::from_graph(graph),
            s: Vec::with_capacity(graph.len()),
            graph,
        }
    }

    /// Returns the index of a node in S
    pub fn get_index_of_node(&self, node: Node) -> Option<usize> {
        if self.not_in_s.contains(&node) {
            None
        } else {
            self.s.iter().position(|&n| n == node)
        }
    }

    /// Inserts a node into S at the specified position and removes conflicting neighbors from S
    pub fn perform_move(&mut self, topo_move: TopoMove) {
        debug_assert!(self.not_in_s.contains(&topo_move.node));

        let node = topo_move.node;
        let mut position = topo_move.position;

        // sort neighbors descending by index so that we can easily remove the neighbors
        let mut incompatible_neighbors = topo_move.incompatible_neighbors;
        incompatible_neighbors.sort_unstable_by(|(_, i_1), (_, i_2)| i_2.cmp(i_1));

        for (neighbor, index_of_neighbor) in incompatible_neighbors {
            debug_assert!(!self.not_in_s.contains(&neighbor));

            self.s.remove(index_of_neighbor);
            self.not_in_s.insert(neighbor);

            // shift position of new node if neighbors to the left of position are removed
            if index_of_neighbor < position {
                position -= 1;
            }
        }

        self.s.insert(position, node);
        self.not_in_s.remove(&node);
    }

    /// Returns all nodes that are conflicting with the passed in move and would be removed from S
    /// if the move would be executed
    pub fn get_conflicting_neighbors(
        &self,
        node: Node,
        position: usize,
    ) -> impl Iterator<Item = (Node, usize)> {
        debug_assert!(self.not_in_s.contains(&node));
        let capacity = self.graph.total_degree(node) as usize;
        let mut incompatible_neighbors =
            FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default());

        // out neighbors
        incompatible_neighbors.extend(
            self.graph
                .out_neighbors(node)
                .map(|u| (u, self.get_index_of_node(u)))
                .filter(|(_, i)| i.is_some())
                .map(|(u, i)| (u, i.unwrap()))
                .filter(|(_, i)| i < &position),
        );

        // in neighbors
        incompatible_neighbors.extend(
            self.graph
                .in_neighbors(node)
                .map(|u| (u, self.get_index_of_node(u)))
                .filter(|(_, i)| i.is_some())
                .map(|(u, i)| (u, i.unwrap()))
                .filter(|(_, i)| i >= &position),
        );

        incompatible_neighbors.into_iter()
    }

    /// Creates a move and calculates which neighbors conflict with it
    pub fn create_move(&self, node: Node, position: usize) -> TopoMove {
        TopoMove {
            node,
            position,
            incompatible_neighbors: self.get_conflicting_neighbors(node, position).collect(),
        }
    }

    /// Returns the feedback vertex set
    pub fn get_fvs(&self) -> Vec<Node> {
        self.not_in_s.cloned_into_vec()
    }
}

/// Represents the move of one node into a topological configuration
pub struct TopoMove {
    pub node: Node,
    pub position: usize,

    /// Contains all neighbors of *node* which have to be removed from the topological configuration
    /// if *node* is added to it. Each tuple stores the neighbor as well as its index inside the
    /// topological configuration
    pub incompatible_neighbors: Vec<(Node, usize)>,
}

#[cfg(test)]
mod tests_topo_config {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;

    #[test]
    fn test_get_index_of_node() {
        let graph = AdjArrayIn::from(&vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
        let mut topo_config = TopologicalConfig::new(&graph);

        for node in 0..5 {
            let topo_move = topo_config.create_move(node, node as usize);
            topo_config.perform_move(topo_move);

            assert_eq!(node as usize, topo_config.get_index_of_node(node).unwrap());
        }
    }

    #[test]
    fn test_get_conflicting_neighbors() {
        let graph = AdjArrayIn::from(&vec![(0, 1), (1, 0)]);
        let mut topo_config = TopologicalConfig::new(&graph);

        let topo_move = topo_config.create_move(0, 0);
        topo_config.perform_move(topo_move);

        let conflicting_neighbors: Vec<_> = topo_config.get_conflicting_neighbors(1, 0).collect();
        assert_eq!(vec![(0, 0)], conflicting_neighbors);
    }

    #[test]
    fn test_perform_move() {
        let graph = AdjArrayIn::from(&vec![(0, 1), (1, 0)]);
        let mut topo_config = TopologicalConfig::new(&graph);

        let mut topo_move = topo_config.create_move(0, 0);
        topo_config.perform_move(topo_move);

        topo_move = topo_config.create_move(1, 0);
        topo_config.perform_move(topo_move);

        assert_eq!(vec![1], topo_config.s);
    }
}
