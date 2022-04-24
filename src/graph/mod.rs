pub mod adj_array;
pub mod adj_array_undir;
pub mod adj_list_matrix;
pub mod connectivity;
pub mod generators;
pub(super) mod graph_macros;
pub mod hash_graph;
pub mod io;
pub mod matrix;
pub mod network_flow;
pub mod node_mapper;
pub mod partition;
pub mod subgraph;
pub mod traversal;
pub mod unique_node_stack;

pub type Node = u32;
pub type Edge = (Node, Node);

pub use adj_array::{AdjArray, AdjArrayIn};
pub use adj_array_undir::AdjArrayUndir;
pub use adj_list_matrix::{AdjListMatrix, AdjListMatrixIn};
pub use connectivity::Connectivity;
pub use io::*;
pub use node_mapper::{Compose, Getter, Inverse, NodeMapper, RankingForwardMapper, Setter};
pub use partition::*;
use std::iter::FromIterator;
pub use subgraph::*;
pub use traversal::*;
pub use unique_node_stack::UniqueNodeStack;

#[cfg(feature = "pace-digest")]
pub mod digest;

#[cfg(feature = "pace-digest")]
pub use self::digest::GraphDigest;

use fxhash::FxHashSet;
use std::ops::Range;

/// Provides getters pertaining to the size of a graph
pub trait GraphOrder {
    type VertexIter<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Returns the number of nodes of the graph
    fn number_of_nodes(&self) -> Node;

    /// Returns the number of edges of the graph
    fn number_of_edges(&self) -> usize;

    /// Return the number of nodes as usize
    fn len(&self) -> usize {
        self.number_of_nodes() as usize
    }

    /// Returns an iterator over V.
    fn vertices(&self) -> Self::VertexIter<'_>;

    /// Returns a range of vertices possibly including deleted vertices
    /// In contrast to self.vertices(), the range returned by self.vertices_ranges() does
    /// not borrow self and hence may be used where additional mutable references of self are needed
    ///
    /// # Warning
    /// This method may iterate over deleted vertices (if supported by an implementation). It is the
    /// responsibility of the caller to identify and treat them accordingly.
    fn vertices_range(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    /// Returns true if the graph has no nodes (and thus no edges)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait AdjacencyListIn: AdjacencyList {
    type IterIn<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    fn in_neighbors(&self, u: Node) -> Self::IterIn<'_>;

    fn in_degree(&self, u: Node) -> Node;

    fn total_degree(&self, u: Node) -> Node {
        self.in_degree(u) + self.out_degree(u)
    }
}

pub trait AdjacencyListUndir: AdjacencyList {
    type IterUndir<'a>: Iterator<Item = Node>
    where
        Self: 'a;
    type IterOutOnly<'a>: Iterator<Item = Node>
    where
        Self: 'a;
    type IterInOnly<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    fn undir_neighbors(&self, u: Node) -> Self::IterUndir<'_>;
    fn undir_degree(&self, u: Node) -> Node;

    fn out_only_neighbors(&self, u: Node) -> Self::IterOutOnly<'_>;
    fn in_only_neighbors(&self, u: Node) -> Self::IterInOnly<'_>;
}

/// Provides basic read-only functionality associated with an adjacency list
pub trait AdjacencyList: GraphOrder + Sized {
    type Iter<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Returns a slice over the outgoing neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn out_neighbors(&self, u: Node) -> Self::Iter<'_>;

    /// Returns the number of outgoing edges from *u*
    fn out_degree(&self, u: Node) -> Node;

    /// Returns an iterator over all edges in the graph in increasing order.
    fn edges_iter(&self) -> EdgeIterator<Self> {
        let mut vertices: Vec<Node> = self.vertices().collect();
        vertices.sort_by(|a, b| b.cmp(a)); // reverse order since we pop from the back
        EdgeIterator {
            graph: self,
            current_node: 0,
            vertices,
            neighbors: Vec::new(),
        }
    }

    /// Returns a vector over all edges in the graph in increasing order.
    /// If the sequence is only scanned once, [`AdjacencyList::edges_iter`] should preferred.
    fn edges_vec(&self) -> Vec<Edge> {
        self.edges_iter().collect()
    }
}

/// Iterates over the edges of a graph in sorted order
pub struct EdgeIterator<'a, G>
where
    G: 'a + AdjacencyList,
{
    graph: &'a G,
    current_node: Node,
    vertices: Vec<Node>,
    neighbors: Vec<Node>,
}

impl<'a, G> Iterator for EdgeIterator<'a, G>
where
    G: 'a + AdjacencyList,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(v) = self.neighbors.pop() {
                return Some((self.current_node, v));
            }

            self.current_node = self.vertices.pop()?;
            self.neighbors = self.graph.out_neighbors(self.current_node).collect();
            self.neighbors.sort_by(|a, b| b.cmp(a)); // reverse order, since we pop from the back
        }
    }
}

/// Provides efficient tests whether an edge exists
pub trait AdjacencyTest {
    /// Returns *true* exactly if the graph contains the directed edge (u, v)
    fn has_edge(&self, u: Node, v: Node) -> bool;

    /// Returns *true* exactly if the graph contains at least one self loop
    fn has_self_loop(&self) -> bool
    where
        Self: GraphOrder,
    {
        self.vertices().any(|u| self.has_edge(u, u))
    }
}

/// Provides efficient tests whether an undirected edge exists
pub trait AdjacencyTestUndir {
    /// Returns *true* exactly if the graph contains the edges (u, v) and (v, u)
    fn has_undir_edge(&self, u: Node, v: Node) -> bool;
}

/// Provides constructor for a forest of isolated nodes
pub trait GraphNew: Default {
    /// Creates an empty graph with n singleton nodes
    fn new(n: usize) -> Self;
}

/// Provides the ability to remove vertices and all incident edges from the graph
pub trait GraphVertexEditing {
    fn remove_vertex(&mut self, u: Node);
    fn has_vertex(&self, u: Node) -> bool;
}

pub trait GraphFromSlice {
    /// Creates a graph with n nodes containing the edges `edges`. This functionally equivalent
    /// to calling `Graph::from(edges)` but may be implement faster. If the edge list is known
    /// to contain unique values, `known_unique` should be set to true for better performance.
    ///
    /// # Warning
    /// If the edge list contains duplicates and `known_unique` is false, the implementation
    /// will work correctly but may or may not allocate too much memory
    fn from_slice(n: Node, edges: &[(Node, Node)], known_unique: bool) -> Self;
}

/// Provides functions to insert/delete edges
pub trait GraphEdgeEditing: GraphNew {
    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is already contained or possibly if u, v >= n **
    fn add_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_add_edge(u, v));
    }

    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// Returns *true* exactly if the edge was not present previously.
    /// ** Can panic if u, v >= n, depending on implementation **
    fn try_add_edge(&mut self, u: Node, v: Node) -> bool;

    /// Adds all edges in the collection
    fn add_edges<'a, T: IntoIterator<Item = &'a Edge>>(&'a mut self, edges: T) {
        for e in edges {
            self.add_edge(e.0, e.1);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    fn remove_edge(&mut self, u: Node, v: Node) {
        assert!(self.try_remove_edge(u, v));
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// If the edge was removed, returns *true* and *false* otherwise.
    /// ** Panics if u, v >= n **
    fn try_remove_edge(&mut self, u: Node, v: Node) -> bool;

    /// Removes all edges into and out of node u
    fn remove_edges_at_node(&mut self, u: Node) {
        self.remove_edges_out_of_node(u);
        self.remove_edges_into_node(u);
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node);

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node);

    /// Removes all edges of the passed in nodes
    fn remove_edges_of_nodes<'a>(&mut self, nodes: impl IntoIterator<Item = &'a Node>)
    where
        Self: AdjacencyListIn,
    {
        for node in nodes {
            self.remove_edges_at_node(*node);
        }
    }

    /// Removes all edges into and out of node `u` and connects every in-neighbor with every out-neighbor.
    /// Returns all nodes that got a self-loop during the process.
    fn contract_node(&mut self, u: Node) -> Vec<Node>;
}
