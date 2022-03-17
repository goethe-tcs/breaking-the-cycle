use crate::bitset::BitSet;
use crate::graph::*;
use fxhash::{FxBuildHasher, FxHashSet};

pub trait TopoGraph: GraphNew + GraphEdgeEditing + AdjacencyListIn + GraphOrder + 'static {}

impl<G> TopoGraph for G where G: GraphNew + GraphEdgeEditing + AdjacencyListIn + GraphOrder + 'static
{}

/// Represents a topological sorting
pub trait TopoConfig<'a, G>
where
    G: TopoGraph,
{
    /// **Assumes that `topo_order` is in topological order! Use [Self::set_state_from_fvs] if this
    /// is not the case.**
    ///
    /// Clears the current topological order and sets it to the passed in one. Doesn't check if the
    /// passed in order is valid!
    fn set_state<I>(&mut self, topo_order: I, fvs: I)
    where
        I: IntoIterator<Item = Node>;

    /// Inserts a node into S at the specified position and removes conflicting neighbors from S
    fn perform_move(&mut self, topo_move: TopoMove);

    /// Returns a reference of the graph for which this configuration was created
    fn graph(&self) -> &'a G;

    /// Returns the index of a node or `None` if the node is not in configuration
    fn get_index(&self, value: &Node) -> Option<usize>;

    /// Returns the amount of nodes that are in the configuration
    fn len(&self) -> usize;

    /// Returns a slice that contains all nodes of the collection
    fn as_slice(&self) -> &[Node];

    /// Returns a slice that contains all nodes of the graph that are not in the configuration
    fn fvs(&self) -> &[Node];

    /// Clears the current topological order, creates a subgraph of the [Self::graph()] using the
    /// inverse of the fvs and creates a topological order for its nodes. Then updates its state
    /// accordingly
    fn set_state_from_fvs<I>(&mut self, fvs: I)
    where
        I: IntoIterator<Item = Node> + Clone,
    {
        let (subgraph, node_mapper) = self
            .graph()
            .vertex_induced(&BitSet::new_all_set_but(self.graph().len(), fvs.clone()));
        let topo_order = node_mapper.get_old_ids(subgraph.topo_search());
        self.set_state(topo_order, fvs.into_iter().collect());
    }

    /// Returns all nodes that are conflicting with the passed in move and would be removed from S
    /// if the move would be executed
    fn get_conflicting_neighbors(&self, node: Node, position: usize) -> Vec<(Node, usize)> {
        debug_assert!(self.fvs().contains(&node));
        let capacity = self.graph().total_degree(node) as usize;
        let mut incompatible_neighbors =
            FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default());

        // out neighbors
        incompatible_neighbors.extend(
            self.graph()
                .out_neighbors(node)
                .map(|u| (u, self.get_index(&u)))
                .filter(|(_, i)| i.is_some())
                .map(|(u, i)| (u, i.unwrap()))
                .filter(|(_, i)| i < &position),
        );

        // in neighbors
        incompatible_neighbors.extend(
            self.graph()
                .in_neighbors(node)
                .map(|u| (u, self.get_index(&u)))
                .filter(|(_, i)| i.is_some())
                .map(|(u, i)| (u, i.unwrap()))
                .filter(|(_, i)| i >= &position),
        );

        incompatible_neighbors.into_iter().collect()
    }

    /// Creates a move and calculates which neighbors conflict with it
    fn create_move(&self, node: Node, position: usize) -> TopoMove {
        TopoMove {
            node,
            position,
            incompatible_neighbors: self.get_conflicting_neighbors(node, position),
        }
    }

    /// Get the left most index-candidate in S without creating a conflict with in-neighbors
    /// which are already in S
    fn get_i_minus_index(&self, node: Node) -> (usize, Vec<usize>) {
        let neighbor_indices: Vec<usize> = self
            .graph()
            .in_neighbors(node)
            .filter_map(|u| self.get_index(&u))
            .collect();

        let pos_i_minus = neighbor_indices.iter().max().map_or(0, |x| x + 1);

        (pos_i_minus, neighbor_indices)
    }

    /// Get the right most index-candidate in S without creating a conflict with out-neighbors
    /// which are already in S
    fn get_i_plus_index(&self, node: Node) -> (usize, Vec<usize>) {
        let neighbor_indices: Vec<usize> = self
            .graph()
            .out_neighbors(node)
            .filter_map(|u| self.get_index(&u))
            .collect();

        let pos_i_plus = neighbor_indices.iter().min().map_or(self.len(), |x| *x);

        (pos_i_plus, neighbor_indices)
    }

    /// R
    fn get_i_index(&self, node: Node) -> usize {
        let (pos_i_minus, in_neighbour) = self.get_i_minus_index(node);
        let (pos_i_plus, out_neighbour) = self.get_i_plus_index(node);
        if in_neighbour > out_neighbour {
            pos_i_minus
        } else {
            pos_i_plus
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether a node is in the configuration or not
    fn contains(&self, value: &Node) -> bool {
        self.get_index(value).is_some()
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

/// Encapsulates the functionality of deciding about which vertex to add to a topological
/// configuration and at which position.
pub trait TopoMoveStrategy<G>
where
    G: TopoGraph,
{
    /// Returns which node should be added to the topological configuration and at which position
    fn next_move<'a, T>(&mut self, topo_config: &T) -> Option<TopoMove>
    where
        T: TopoConfig<'a, G>;

    /// This method is called whenever a move is performed. May be used by the `TopoMoveStrategy`
    /// to update its state accordingly.
    fn on_before_perform_move<'a, T>(&mut self, _topo_config: &T, _topo_move: &TopoMove)
    where
        T: TopoConfig<'a, G>,
    {
    }

    /// This method is called whenever a move is rejected. May be used by the `TopoMoveStrategy`
    /// to update its state accordingly.
    fn on_move_rejected<'a, T>(&mut self, _topo_config: &T, _topo_move: &TopoMove)
    where
        T: TopoConfig<'a, G>,
    {
    }
}
