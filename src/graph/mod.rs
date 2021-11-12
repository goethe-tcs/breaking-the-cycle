pub mod adj_list_matrix;
pub mod connectivity;
pub mod io;

use std::ops::Range;

pub type Node = u32;
pub type Edge = (Node, Node);

pub use adj_list_matrix::AdjListMatrix;
pub use connectivity::Connectivity;

pub trait GraphOrder {
    /// Returns the order (number of nodes) of the graph
    fn order(&self) -> Node;

    /// Return the number of nodes as usize
    fn len(&self) -> usize {self.order() as usize}

    /// Returns an iterator over V.
    fn vertices(&self) -> Range<Node> {
        0..self.order()
    }
}

///! Provides basic read-only functionality associated with an adjecency list
pub trait AdjecencyList : GraphOrder {
    /// Returns a slice over the outgoing neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn out_neighbors(&self, u: Node) -> &[Node];

    /// Returns a slice over the incoming neighbors of a given vertex.
    /// ** Panics if the v >= n **
    fn in_neighbors(&self, u: Node) -> &[Node];

    /// Returns the number of ingoing edges to *u*
    fn in_degree(&self, u: Node) -> Node {
        self.in_neighbors(u).len() as Node
    }

    /// Returns the number of outgoing edges from *u*
    fn out_degree(&self, u: Node) -> Node { self.out_neighbors(u).len() as Node }

    /// Returns the total number of edges incident to *u*
    fn total_degree(&self, u: Node) -> Node {
        self.in_degree(u) + self.out_degree(u)
    }

    /// Returns a vector over all edges in the graph
    fn edges(&self) -> Vec<Edge> {
        self.vertices().map(|u| {
            self.out_neighbors(u).iter().map(move |&v| {(u, v)})
        }).flatten().collect()
    }
}

pub trait AdjecencyTest {
    /// Returns *true* exactly if the graph contains the directed edge (u, v)
    fn has_edge(&self, u: Node, v: Node) -> bool;
}

pub trait GraphManipulation {
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
}

pub trait GraphNew {
    /// Creates an empty graph with n singleton nodes
    fn new(n: usize) -> Self;
}

