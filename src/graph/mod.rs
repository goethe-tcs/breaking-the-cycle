pub mod adj_list_matrix;
pub mod connectivity;
pub mod io;
pub mod traversal;

use std::ops::Range;

pub type Node = u32;
pub type Edge = (Node, Node);

pub use adj_list_matrix::AdjListMatrix;
pub use connectivity::Connectivity;
pub use traversal::*;

/// Provides getters pertaining to the size of a graph
pub trait GraphOrder {
    /// Returns the number of nodes of the graph
    fn number_of_nodes(&self) -> Node;

    /// Returns the number of edges of the graph
    fn number_of_edges(&self) -> usize;

    /// Return the number of nodes as usize
    fn len(&self) -> usize {self.number_of_nodes() as usize}

    /// Returns an iterator over V.
    fn vertices(&self) -> Range<Node> {
        0..self.number_of_nodes()
    }

    /// Returns true if the graph has no nodes (and thus no edges)
    fn is_empty(&self) -> bool {self.len() == 0}
}

/// Provides basic read-only functionality associated with an adjacency list
pub trait AdjacencyList: GraphOrder {
    type Iter<'a>: Iterator<Item = Node> where Self: 'a;

    /// Returns a slice over the outgoing neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn out_neighbors(&self, u: Node) -> Self::Iter<'_>;

    /// Returns a slice over the incoming neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn in_neighbors(&self, u: Node) -> Self::Iter<'_>;

    /// Returns the number of ingoing edges to *u*
    fn in_degree(&self, u: Node) -> Node;

    /// Returns the number of outgoing edges from *u*
    fn out_degree(&self, u: Node) -> Node;

    /// Returns the total number of edges incident to *u*
    fn total_degree(&self, u: Node) -> Node {
        self.in_degree(u) + self.out_degree(u)
    }

    /// Returns a vector over all edges in the graph
    fn edges(&self) -> Vec<Edge> {
        self.vertices().map(|u| {
            self.out_neighbors(u).map(move |v| {(u, v)})
        }).flatten().collect()
    }
}

/// Provides efficient tests whether an edge exists
pub trait AdjacencyTest {
    /// Returns *true* exactly if the graph contains the directed edge (u, v)
    fn has_edge(&self, u: Node, v: Node) -> bool;
}

/// Provides constructor for a forest of isolated nodes
pub trait GraphNew {
    /// Creates an empty graph with n singleton nodes
    fn new(n: usize) -> Self;
}

/// Provides functions to insert/delete edges
pub trait GraphEdgeEditing : GraphNew {
    /// Adds the directed edge *(u,v)* to the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is already contained or u, v >= n **
    fn add_edge(&mut self, u: Node, v: Node);

    /// Adds all edges in the collection
    fn add_edges<'a, T: IntoIterator<Item = &'a Edge>>(&'a mut self, edges: T) {
        for e in edges {
            self.add_edge(e.0, e.1);
        }
    }

    /// Removes the directed edge *(u,v)* from the graph. I.e., the edge FROM u TO v.
    /// ** Panics if the edge is not present or u, v >= n **
    fn remove_edge(&mut self, u: Node, v: Node);

    /// Removes all edges into and out of node u
    fn remove_edges_at_node(&mut self, u: Node) {
        self.remove_edges_out_of_node(u);
        self.remove_edges_into_node(u);
    }

    /// Removes all edges into node u, i.e. post-condition the in-degree is 0
    fn remove_edges_into_node(&mut self, u: Node);

    /// Removes all edges out of node u, i.e. post-condition the out-degree is 0
    fn remove_edges_out_of_node(&mut self, u: Node);

}