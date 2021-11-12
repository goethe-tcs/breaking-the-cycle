use crate::bitset::BitSet;
use crate::graph::Graph;
use std::collections::VecDeque;

pub struct EdmondsKarp {
    residual_network: ResidualNetwork,
    predecessor: Vec<u32>,
}

pub struct ResidualNetwork {
    s: u32,
    t: u32,
    n: usize,
    capacity: Vec<BitSet>,
    labels: Vec<u32>,
}

impl ResidualNetwork {
    pub fn reverse(&mut self, u: usize, v: usize) {
        assert!(self.capacity[u][v]);
        self.capacity[u].unset_bit(v);
        self.capacity[v].set_bit(u);
    }

    pub fn for_edge_disjoint(graph: &Graph, s: u32, t: u32) -> Self {
        let n = graph.order() as usize;
        Self {
            s,
            t,
            n,
            capacity: graph
                .vertices()
                .map(|u| {
                    let mut bs = BitSet::new(n);
                    for v in graph.vertices() {
                        if graph.has_edge(u, v) {
                            bs.set_bit(v as usize);
                        }
                    }
                    bs
                })
                .collect(),
            labels: graph.vertices().collect(),
        }
    }

    // To find vertex disjoint paths the graph must be transformed:
    // for each vertex v create two vertices v_in and v_out.
    // for each original edge (u, v) fcreate the edge (u_out, v_in)
    // for each v_in add the edge (v_in, v_out).
    //
    // Ignore this procedure for the vertices s and t. For those simply
    // add the edges (s, v_in) for each edge (s, v) in the original graph
    // and the edge (v_out, s) for each edge (v, s) in the original graph.
    pub fn for_vertex_disjoint(graph: &Graph, s: u32, t: u32) -> Self {
        let n = graph.order() as usize * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![BitSet::new(n); n];
        for v in graph.vertices() {
            // handle s and t
            if v == s || v == t {
                for u in graph.out_neighbors(v) {
                    let u_in = *u as usize;
                    // add edge from v to u_in
                    capacity[v as usize].set_bit(u_in);
                }
                continue;
            }

            let v_in = v as usize;
            let v_out = graph.order() as usize + v as usize;
            // from v_in to v_out
            capacity[v_in].set_bit(v_out);

            for u in graph.out_neighbors(v) {
                // this also handles s and t
                let u_in = *u as usize;
                // add edge from v_out to u_in
                capacity[v_out].set_bit(u_in);
            }
        }

        Self {
            s,
            t,
            n,
            capacity,
            labels,
        }
    }

    // creates network for finding vertex disjoint cycles that all share the vertex s. the
    // same as vertex disjoint, but s is handled the same as all other vertices, except that
    // s_in and s_out are not connected. Then, s_out is the source and s_in is the target.
    fn for_petals(graph: &Graph, s: u32) -> Self {
        let n = graph.order() as usize * 2; // duplicate
        let labels: Vec<_> = graph.vertices().chain(graph.vertices()).collect();

        let mut capacity = vec![BitSet::new(n); n];
        for v in graph.vertices() {
            // handle s and t
            let v_in = v as usize;
            let v_out = graph.order() as usize + v as usize;
            // from v_in to v_out. Unless the vertex is s.
            if v != s {
                capacity[v_in].set_bit(v_out);
            }

            for u in graph.out_neighbors(v) {
                let u_in = *u as usize;
                // add edge from v_out to u_in
                capacity[v_out].set_bit(u_in);
            }
        }

        Self {
            s: graph.order() + s,
            t: s,
            n,
            capacity,
            labels,
        }
    }
}

impl EdmondsKarp {
    pub fn new(residual_network: ResidualNetwork) -> Self {
        let n = residual_network.n;
        Self {
            residual_network,
            predecessor: vec![0; n],
        }
    }

    fn bfs(&mut self) -> bool {
        let n = self.residual_network.n;
        let s = self.residual_network.s;
        let t = self.residual_network.t;
        let mut visited = BitSet::new(n);
        let mut q: VecDeque<_> = Default::default();
        self.predecessor[s as usize] = u32::MAX;
        q.push_back(s);
        visited.set_bit(s as usize);
        while let Some(u) = q.pop_front() {
            for v in 0..n {
                if !visited[v] && self.residual_network.capacity[u as usize][v as usize] {
                    q.push_back(v as u32);
                    self.predecessor[v] = u;
                    visited.set_bit(v);
                }
            }
        }
        visited[t as usize]
    }

    // Finds the number of edge disjoint paths from s to t
    pub fn num_disjoint(&mut self) -> u32 {
        self.map(|_| 1).sum()
    }

    // Outputs all edge disjoint paths from s to t. The paths are vertex disjoint in the original graph
    // when the network has been constructed for said restriction.
    // Each path is represented as a vector of vertices
    pub fn disjoint_paths(&mut self) -> Vec<Vec<u32>> {
        self.collect()
    }
}

impl Iterator for EdmondsKarp {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.bfs() {
            return None;
        }

        let t = self.residual_network.t;
        let s = self.residual_network.s;
        let mut path = vec![t];
        let mut v = t;
        while v != s {
            let u = self.predecessor[v as usize];
            // when trying to find vertex disjoint this skips edges inside 'gadgets'
            if self.residual_network.labels[u as usize] != self.residual_network.labels[v as usize]
            {
                path.push(u);
            }
            self.residual_network.reverse(u as usize, v as usize);
            v = u;
        }
        Some(
            path.iter()
                .map(|v| self.residual_network.labels[*v as usize])
                .rev()
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;
    use crate::network_flow::{EdmondsKarp, ResidualNetwork};

    const EDGES: [(u32, u32); 13] = [
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
        let mut g = Graph::new(8);
        g.add_edges(&EDGES);
        let edges_reverse: Vec<_> = EDGES.iter().map(|(u, v)| (*v, *u)).collect();
        g.add_edges(&edges_reverse);
        let mut ec = EdmondsKarp::new(ResidualNetwork::for_edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 3);
    }

    #[test]
    fn edmonds_carp_vertex_disjoint() {
        let mut g = Graph::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualNetwork::for_vertex_disjoint(&g, 0, 7));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 1);
    }

    #[test]
    fn edmonds_carp_petals() {
        let mut g = Graph::new(8);
        g.add_edges(&EDGES);
        g.add_edge(3, 7);
        g.add_edge(7, 0);
        let mut ec = EdmondsKarp::new(ResidualNetwork::for_petals(&g, 3));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 2);
    }

    #[test]
    fn edmonds_carp_no_co_arcs() {
        let mut g = Graph::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualNetwork::for_edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 2);
    }
}
