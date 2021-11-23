use crate::bitset::BitSet;
use crate::graph::{AdjacencyList, GraphOrder, Node, Traversal, TraversalState};

pub struct EdmondsKarp {
    residual_network: ResidualBitMatrix,
    predecessor: Vec<Node>,
}

trait ResidualNetwork: SourceTarget + AdjacencyList + Label<Node> {
    /// Reverses the edge (u, v) to (v, u).
    fn reverse(&mut self, u: Node, v: Node);

    /// Constructs a network to find edge-disjoint paths from s to t
    fn edge_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self;

    /// To find vertex disjoint paths the graph must be transformed:
    /// for each vertex v create two vertices v_in and v_out.
    /// for each original edge (u, v) create the edge (u_out, v_in)
    /// for each v_in add the edge (v_in, v_out).
    ///
    /// Ignore this procedure for the vertices s and t. For those simply
    /// add the edges (s, v_in) for each edge (s, v) in the original graph
    /// and the edge (v_out, s) for each edge (v, s) in the original graph.
    fn vertex_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self;

    /// creates network for finding vertex disjoint cycles that all share the vertex s. the
    /// same as vertex disjoint, but s is handled the same as all other vertices, except that
    /// s_in and s_out are not connected. Then, s_out is the source and s_in is the target.
    fn petals<G: GraphOrder + AdjacencyList>(graph: &G, s: Node) -> Self;
}

pub struct ResidualBitMatrix {
    s: Node,
    t: Node,
    n: usize,
    m: usize,
    capacity: Vec<BitSet>,
    labels: Vec<Node>,
}

trait Label<T> {
    fn label(&self, u: Node) -> &T;
}

impl Label<Node> for ResidualBitMatrix {
    fn label(&self, u: Node) -> &Node {
        &self.labels[u as usize]
    }
}

trait SourceTarget {
    fn source(&self) -> &Node;
    fn target(&self) -> &Node;
    fn source_mut(&mut self) -> &mut Node;
    fn target_mut(&mut self) -> &mut Node;
}

impl SourceTarget for ResidualBitMatrix {
    fn source(&self) -> &Node {
        &self.s
    }

    fn target(&self) -> &Node {
        &self.t
    }

    fn source_mut(&mut self) -> &mut Node {
        &mut self.t
    }

    fn target_mut(&mut self) -> &mut Node {
        &mut self.t
    }
}

impl GraphOrder for ResidualBitMatrix {
    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }

    fn number_of_edges(&self) -> usize {
        self.m
    }
}

impl AdjacencyList for ResidualBitMatrix {
    type Iter<'a> = impl Iterator<Item = Node> + 'a;

    fn out_neighbors(&self, u: Node) -> Self::Iter<'_> {
        self.capacity[u as usize].iter().map(|u| u as Node)
    }

    fn out_degree(&self, u: Node) -> Node {
        self.capacity[u as usize].cardinality() as Node
    }
}

impl ResidualNetwork for ResidualBitMatrix {
    fn reverse(&mut self, u: Node, v: Node) {
        let u = u as usize;
        let v = v as usize;
        assert!(self.capacity[u][v]);
        self.capacity[u].unset_bit(v);
        self.capacity[v].set_bit(u);
    }

    fn edge_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self {
        let n = graph.len();
        Self {
            s,
            t,
            n,
            m: 0,
            capacity: graph
                .vertices()
                .map(|u| BitSet::new_all_unset_but(n, graph.out_neighbors(u)))
                .collect(),
            labels: graph.vertices().collect(),
        }
    }

    fn vertex_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self {
        let n = graph.len() * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![BitSet::new(n); n];
        for v in graph.vertices() {
            // handle s and t
            if v == s || v == t {
                for u in graph.out_neighbors(v) {
                    let u_in = u as usize;
                    // add edge from v to u_in
                    capacity[v as usize].set_bit(u_in);
                }
                continue;
            }

            let v_in = v as usize;
            let v_out = graph.len() + v as usize;
            // from v_in to v_out
            capacity[v_in].set_bit(v_out);

            for u in graph.out_neighbors(v) {
                // this also handles s and t
                let u_in = u as usize;
                // add edge from v_out to u_in
                capacity[v_out].set_bit(u_in);
            }
        }

        Self {
            s,
            t,
            n,
            m: 0,
            capacity,
            labels,
        }
    }

    fn petals<G: GraphOrder + AdjacencyList>(graph: &G, s: Node) -> Self {
        let n = graph.len() * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![BitSet::new(n); n];
        for v in graph.vertices() {
            // handle s and t
            let v_in = v as usize;
            let v_out = graph.len() + v as usize;
            // from v_in to v_out. Unless the vertex is s.
            if v != s {
                capacity[v_in].set_bit(v_out);
            }

            for u in graph.out_neighbors(v) {
                let u_in = u as usize;
                // add edge from v_out to u_in
                capacity[v_out].set_bit(u_in);
            }
        }

        Self {
            s: graph.number_of_nodes() + s,
            t: s,
            n,
            m: capacity.iter().map(|v| v.cardinality()).sum(),
            capacity,
            labels,
        }
    }
}

impl EdmondsKarp {
    pub fn new(residual_network: ResidualBitMatrix) -> Self {
        let n = residual_network.len();
        Self {
            residual_network,
            predecessor: vec![0; n],
        }
    }

    fn bfs(&mut self) -> bool {
        let s = *self.residual_network.source();
        let t = *self.residual_network.target() as usize;
        let predecessor = &mut self.predecessor;
        let f = Box::new(|u, v| {
            predecessor[v as usize] = u;
        });
        let mut bfs = self.residual_network.bfs(s).register_pre_push(f);
        loop {
            match bfs.next() {
                None => {
                    return bfs.visited()[t];
                }
                Some(_) => {
                    continue;
                }
            }
        }
    }

    /// Finds the number of edge disjoint paths from s to t
    pub fn num_disjoint(&mut self) -> usize {
        self.count()
    }

    /// Outputs all edge disjoint paths from s to t. The paths are vertex disjoint in the original graph
    /// when the network has been constructed for said restriction.
    /// Each path is represented as a vector of vertices
    pub fn disjoint_paths(&mut self) -> Vec<Vec<Node>> {
        self.collect()
    }
}

impl Iterator for EdmondsKarp {
    type Item = Vec<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.bfs() {
            return None;
        }

        let t = *self.residual_network.target();
        let s = *self.residual_network.source();
        let mut path = vec![t];
        let mut v = t;
        while v != s {
            let u = self.predecessor[v as usize];
            // when trying to find vertex disjoint this skips edges inside 'gadgets'
            if self.residual_network.label(u) != self.residual_network.label(v) {
                path.push(u);
            }
            self.residual_network.reverse(u, v);
            v = u;
        }
        Some(
            path.iter()
                .map(|v| *self.residual_network.label(*v))
                .rev()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::network_flow::{EdmondsKarp, ResidualBitMatrix, ResidualNetwork};
    use crate::graph::{AdjListMatrix, GraphEdgeEditing, GraphNew, Node};

    const EDGES: [(Node, Node); 13] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
        (2, 6),
        (3, 6),
        (4, 2),
        (4, 7),
        (5, 1),
        (5, 7),
        (6, 7),
        (6, 5),
    ];

    #[test]
    fn edmonds_carp() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let edges_reverse: Vec<_> = EDGES.iter().map(|(u, v)| (*v, *u)).collect();
        g.add_edges(&edges_reverse);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 3);
    }

    #[test]
    fn edmonds_carp_vertex_disjoint() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::vertex_disjoint(&g, 0, 7));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 1);
    }

    #[test]
    fn edmonds_carp_petals() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        g.add_edge(3, 7);
        g.add_edge(7, 0);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 2);
    }

    #[test]
    fn edmonds_carp_no_co_arcs() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 2);
    }
}
