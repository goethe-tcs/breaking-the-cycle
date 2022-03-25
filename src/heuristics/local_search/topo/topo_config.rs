use crate::bitset::BitSet;
use crate::graph::*;
use crate::heuristics::local_search::topo::topo_config::MovePosition::Index;
use fxhash::{FxBuildHasher, FxHashSet};
use std::fmt::Debug;

pub trait TopoGraph:
    GraphNew + GraphEdgeEditing + AdjacencyListIn + GraphOrder + Clone + Debug + 'static
{
}

impl<G> TopoGraph for G where
    G: GraphNew + GraphEdgeEditing + AdjacencyListIn + GraphOrder + Clone + Debug + 'static
{
}

/// Represents a topological sorting
pub trait TopoConfig<'a, G>
where
    G: TopoGraph,
    Self: Sized,
{
    fn new(graph: &'a G) -> Self;

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

    fn new_with_fvs<I>(graph: &'a G, fvs: I) -> Self
    where
        I: IntoIterator<Item = Node> + Clone,
    {
        let mut result = Self::new(graph);
        result.set_state_from_fvs(fvs);
        result
    }

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
    fn calc_conflicts(&self, node: Node, position: MovePosition) -> Vec<(Node, usize)> {
        debug_assert!(self.fvs().contains(&node));
        let capacity = self.graph().total_degree(node) as usize;
        let mut incompatible_neighbors =
            FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default());
        let position = position.resolve(self, node);

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
        TopoMove::new_with_conflicts(node, position, self.calc_conflicts(node, Index(position)))
    }

    /// Creates a move for i- (left tuple result) and a move for i+ (right tuple result) of the node
    fn calc_move_candidates(&self, node: Node) -> (TopoMove, TopoMove) {
        let (i_minus, in_neighbors) = self.get_i_minus_index(node);
        let (i_plus, out_neighbors) = self.get_i_plus_index(node);

        // no conflicts
        if i_minus <= i_plus {
            (
                TopoMove::new_with_conflicts(node, i_minus, vec![]),
                TopoMove::new_with_conflicts(node, i_plus, vec![]),
            )
        }
        // conflicts in range [i_plus..i_minus]
        else {
            // get incompatible out-neighbors for i- by finding all indices smaller than (i-)
            let mut confl_i_minus = Vec::with_capacity(out_neighbors.len());
            confl_i_minus.extend(out_neighbors.into_iter().filter(|(_, i)| i < &i_minus));

            // get incompatible in-neighbors for i+ by finding all indices greater than i+
            let mut confl_i_plus = Vec::with_capacity(in_neighbors.len());
            confl_i_plus.extend(in_neighbors.into_iter().filter(|(_, i)| i >= &i_plus));

            (
                TopoMove::new_with_conflicts(node, i_minus, confl_i_minus),
                TopoMove::new_with_conflicts(node, i_plus, confl_i_plus),
            )
        }
    }

    /// Get the left most index-candidate in S without creating a conflict with in-neighbors
    /// which are already in S.
    ///
    /// Also returns the in-neighbors of the node (the ones that are in the topological
    /// configuration) together with their index.
    fn get_i_minus_index(&self, node: Node) -> (usize, Vec<(Node, usize)>) {
        let neighbors = self
            .graph()
            .in_neighbors(node)
            .filter_map(|u| self.get_index(&u).map(|i| (u, i)))
            .collect::<Vec<_>>();

        let pos_i_minus = neighbors
            .iter()
            .max_by_key(|(_, i)| i)
            .map_or(0, |(_, i)| i + 1);
        (pos_i_minus, neighbors)
    }

    /// Get the right most index-candidate in S without creating a conflict with out-neighbors
    /// which are already in S.
    ///
    /// Also returns the out-neighbors of the node (the ones that are in the topological
    /// configuration) together with their index.
    fn get_i_plus_index(&self, node: Node) -> (usize, Vec<(Node, usize)>) {
        let neighbors = self
            .graph()
            .out_neighbors(node)
            .filter_map(|u| self.get_index(&u).map(|i| (u, i)))
            .collect::<Vec<_>>();

        let pos_i_plus = neighbors
            .iter()
            .min_by_key(|(_, i)| i)
            .map_or(self.len(), |(_, i)| *i);
        (pos_i_plus, neighbors)
    }

    /// Returns the better choice between i+ and i-
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
    /// The node that is moved from the fvs into the topological configuration
    node: Node,

    /// The position in the topological configuration where the node is added
    position: MovePosition,

    /// By how much the size of the fvs (the inverse of the topological configuration) would change
    /// if this move would be performed
    performance: isize,

    /// Contains all neighbors of `node` which have to be removed from the topological configuration
    /// if the move would be performed. Each tuple stores the neighbor as well as its index inside
    /// the topological configuration
    conflicting_nodes: Option<Vec<(Node, usize)>>,
}

impl TopoMove {
    /// Use this constructor if the conflicting neighbors for this move are known
    pub fn new_with_conflicts(
        node: Node,
        position: usize,
        incompatible: Vec<(Node, usize)>,
    ) -> Self {
        Self {
            node,
            position: Index(position),
            performance: (incompatible.len() as isize) - 1,
            conflicting_nodes: Some(incompatible),
        }
    }

    /// Use this constructor if the performance is known (because it is cached somehow) but the
    /// conflicting neighbors are not
    pub fn new_as_cached(node: Node, position: MovePosition, performance: isize) -> Self {
        Self {
            node,
            position,
            conflicting_nodes: None,
            performance,
        }
    }

    pub fn node(&self) -> Node {
        self.node
    }

    pub fn position(&self) -> MovePosition {
        self.position
    }

    pub fn conflicting_nodes(&self) -> Option<&Vec<(Node, usize)>> {
        self.conflicting_nodes.as_ref()
    }

    pub fn performance(&self) -> isize {
        self.performance
    }

    pub fn get_or_calc_conflicts<'a, G, T>(&mut self, topo_config: &T) -> &[(Node, usize)]
    where
        G: TopoGraph,
        T: TopoConfig<'a, G>,
    {
        if self.conflicting_nodes.is_none() {
            self.conflicting_nodes = Some(topo_config.calc_conflicts(self.node, self.position))
        }

        self.conflicting_nodes.as_ref().unwrap().as_slice()
    }

    pub fn consume<'a, G, T>(mut self, topo_config: &T) -> (Node, usize, isize, Vec<(Node, usize)>)
    where
        G: TopoGraph,
        T: TopoConfig<'a, G>,
    {
        let position = self.position.resolve(topo_config, self.node());
        let conflicts = self
            .conflicting_nodes
            .take()
            .unwrap_or_else(|| topo_config.calc_conflicts(self.node(), Index(position)));
        (self.node, position, self.performance, conflicts)
    }
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum MovePosition {
    Index(usize),
    IMinus,
    IPlus,
}

impl MovePosition {
    fn resolve<'a, G, T>(&self, topo_config: &T, node: Node) -> usize
    where
        G: TopoGraph,
        T: TopoConfig<'a, G>,
    {
        match &self {
            Index(position) => *position,
            MovePosition::IMinus => topo_config.get_i_minus_index(node).0,
            MovePosition::IPlus => topo_config.get_i_plus_index(node).0,
        }
    }
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
    fn on_before_perform_move<'a, T>(&mut self, _topo_config: &T, _topo_move: &mut TopoMove)
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
