use crate::bitset::BitSet;
use crate::graph::{AdjacencyList, GraphOrder, Node, Traversal, TraversalState, TraversalTree};
use num::Integer;
use rand::Rng;
use std::collections::HashSet;

pub struct EdmondsKarp {
    residual_network: ResidualBitMatrix,
    predecessor: Vec<Node>,
    changes_on_bitmatrix: Vec<(Node, Node)>,
    remember_changes: bool,
}

pub trait ResidualNetwork: SourceTarget + AdjacencyList + Label<Node> {
    /// Reverses the edge (u, v) to (v, u).
    fn reverse(&mut self, u: Node, v: Node);

    /// Constructs a network to find edge-disjoint paths from s to t
    fn edge_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self;

    /// To find vertex disjoint paths the graph must be transformed:
    /// for each vertex v create two vertices v_in (with index v) and v_out (with index v + n).
    /// for each original edge (u, v) create the edge (u_out, v_in)
    /// for each v_in add the edge (v_in, v_out).
    ///
    /// Ignore this procedure for the vertices s and t. For those simply
    /// add the edges (s, v_in) for each edge (s, v) in the original graph
    /// and the edge (v_out, t) for each edge (v, t) in the original graph.
    fn vertex_disjoint<G: GraphOrder + AdjacencyList>(graph: &G, s: Node, t: Node) -> Self;

    /// creates network for finding vertex disjoint cycles that all share the vertex s. the
    /// same as vertex disjoint, but s is handled the same as all other vertices, except that
    /// s_in and s_out are not connected. Then, s_out is the source and s_in is the target.
    fn petals<G: GraphOrder + AdjacencyList>(graph: &G, s: Node) -> Self;

    /// can be useful if one wants to reuse a capacity many times.
    /// creating a new capacity could be expensive.
    fn from_capacity_and_labels(capacity: Vec<BitSet>, labels: Vec<Node>, s: Node, t: Node)
        -> Self;

    /// While petals get calculated, the capacity changes its bits. undo_changes reverts all these changes.
    fn undo_changes<I: IntoIterator<Item = (Node, Node)>>(&mut self, changes_on_bitmatrix: I);
}

pub struct ResidualBitMatrix {
    s: Node,
    t: Node,
    n: usize,
    m: usize,
    capacity: Vec<BitSet>,
    labels: Vec<Node>,
}

pub trait Label<T> {
    fn label(&self, u: Node) -> &T;
}

impl Label<Node> for ResidualBitMatrix {
    fn label(&self, u: Node) -> &Node {
        &self.labels[u as usize]
    }
}

pub trait SourceTarget {
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
    type VertexIter<'a> = impl Iterator<Item = Node> + 'a;

    fn number_of_nodes(&self) -> Node {
        self.n as Node
    }

    fn number_of_edges(&self) -> usize {
        self.m
    }

    fn vertices(&self) -> Self::VertexIter<'_> {
        self.vertices_range()
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

    fn from_capacity_and_labels(
        capacity: Vec<BitSet>,
        labels: Vec<Node>,
        s: Node,
        t: Node,
    ) -> Self {
        Self {
            s,
            t,
            n: capacity.len(),
            m: capacity.iter().map(|v| v.cardinality()).sum(),
            capacity,
            labels,
        }
    }

    fn undo_changes<I: IntoIterator<Item = (Node, Node)>>(&mut self, changes: I) {
        changes.into_iter().for_each(|(u, v)| self.reverse(v, u));
    }
}

impl ResidualBitMatrix {
    pub fn take(self) -> (Vec<BitSet>, Vec<Node>) {
        (self.capacity, self.labels)
    }
}

impl EdmondsKarp {
    pub fn new(residual_network: ResidualBitMatrix) -> Self {
        let n = residual_network.len();
        Self {
            residual_network,
            predecessor: vec![0; n],
            changes_on_bitmatrix: vec![],
            remember_changes: false,
        }
    }

    fn bfs(&mut self) -> bool {
        let s = *self.residual_network.source();
        let t = *self.residual_network.target();

        let mut bfs = self.residual_network.bfs_with_predecessor(s);
        bfs.stop_at(t);
        bfs.parent_array_into(self.predecessor.as_mut_slice());
        bfs.did_visit_node(t)
    }

    /// Finds the number of edge disjoint paths from s to t
    pub fn num_disjoint(&mut self) -> usize {
        self.count()
    }

    /// Finds the number of edge discoint paths from s to t, but stops counting at a given k.
    pub fn count_num_disjoint_upto(&mut self, k: Node) -> Node {
        self.take(k as usize).count() as Node
    }

    /// Outputs all edge disjoint paths from s to t. The paths are vertex disjoint in the original graph
    /// when the network has been constructed for said restriction.
    /// Each path is represented as a vector of vertices
    pub fn disjoint_paths(&mut self) -> Vec<Vec<Node>> {
        self.collect()
    }

    /// Sets/Unsets remember_changes to true/false. If its true, the changes, that are made on
    /// the capacity, are remembered. With that it is possible to undo these changes later.
    pub fn set_remember_changes(&mut self, remember_changes: bool) {
        self.remember_changes = remember_changes;
    }

    /// reverts each edge in capacity, that got changed while petals were calculated.
    pub fn undo_changes(&mut self) {
        self.residual_network
            .undo_changes(self.changes_on_bitmatrix.iter().rev().copied());
    }

    /// gives access to the capacity, so it can for example be reused for a new ResidualBitMatrix.
    pub fn take(self) -> (Vec<BitSet>, Vec<Node>) {
        self.residual_network.take()
    }
}

pub trait RememberChanges {
    fn remember_changes(&mut self, u: Node, v: Node);
}

impl RememberChanges for EdmondsKarp {
    fn remember_changes(&mut self, u: Node, v: Node) {
        self.changes_on_bitmatrix.push((u, v));
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

            if self.remember_changes {
                self.remember_changes(u, v);
            }

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

pub trait MinVertexCut: AdjacencyList {
    /// Computes a minimum (s, t) vertex cut and returns the number of vertices reachable from and
    /// including `s` after the cut is applied. For performance reasons a maximal acceptable size
    /// (inclusive) may be supplied. The methods returns `None` iff the minimum (s, t) cut size excceds
    /// `max_size`.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::*;
    /// //        /-> 1 -\
    /// // (s) 0 =         => 3 (t)
    /// //        \-> 2 -/
    /// let graph = AdjArray::from(&[(0, 1), (0, 2), (1,3), (2,3)]);
    /// let (cut, size_s) = graph.min_st_vertex_cut(0, 3, None).unwrap();
    /// assert_eq!(size_s, 1); // only s is reachable
    /// assert!(cut == vec![1,2] || cut == vec![2,1]);
    /// ```
    fn min_st_vertex_cut(
        &self,
        s: Node,
        t: Node,
        max_size: Option<Node>,
    ) -> Option<(Vec<Node>, Node)> {
        let max_size = max_size.unwrap_or(self.number_of_nodes() - 2);

        let mut ek = EdmondsKarp::new(ResidualBitMatrix::vertex_disjoint(self, s, t));
        let flow_lower_bound = ek.count_num_disjoint_upto(max_size + 1);
        if flow_lower_bound > max_size {
            return None;
        }

        let mut cut_candidates = HashSet::with_capacity(2 * flow_lower_bound as usize);
        let mut size_s_cut: usize = 2; // count s itself

        let mut dfs = ek.residual_network.dfs(*ek.residual_network.source());
        dfs.next(); // skip source

        for v in dfs.by_ref() {
            debug_assert_ne!(v, *ek.residual_network.target());
            size_s_cut += 1;

            let v = if v < self.number_of_nodes() {
                v
            } else {
                v - self.number_of_nodes()
            };

            if !cut_candidates.insert(v) {
                // already contained: remove!
                cut_candidates.remove(&v);
            }
        }

        size_s_cut -= cut_candidates.len();

        // Add a cut-vertex for isolated s->t paths
        for v in self.out_neighbors(s) {
            // Todo: here we arbitrarily choose a node closest to the source; this might not be a good
            // choice towards a balanced cut. But given the density of our kernels, it should not
            // matter too much

            if dfs.did_visit_node(v) {
                continue;
            }
            if !ek.residual_network.capacity[v as usize][s as usize] {
                continue;
            }
            let success = cut_candidates.insert(v);
            debug_assert!(success);
        }

        assert_eq!(cut_candidates.len(), flow_lower_bound as usize);
        assert!(size_s_cut.is_even());

        Some((
            cut_candidates.iter().copied().collect(),
            (size_s_cut / 2) as Node,
        ))
    }

    /// Approximates a balanced min vertex cut as follows: repeat `attempts` many times:
    /// Randomly select `s` and `t` and compute a minimum (s-t). If the number of nodes reachable
    /// from `s` AND the number of nodes that can reach `t` reach at least `self.len() * imbalance - 1`
    /// declare the cut legal. Then return the small legal cut.
    ///
    /// # Warning
    /// If `imbalance` exceeds 0.5 no solution will be found
    fn approx_min_balanced_cut<R: Rng>(
        &self,
        rng: &mut R,
        attempts: usize,
        imbalance: f64,
        max_size: Option<Node>,
        greedy: bool,
    ) -> Option<Vec<Node>> {
        assert!(self.len() > 2);

        // we are not using (0..attempts). ... .min_by_key(|c| c.len()) since each we keep track
        // of the current best solution and pass it as an upper bound into min_st
        let mut current_upper_bound = max_size;
        let mut current_best = None;

        for _ in 0..attempts {
            let s = rng.gen_range(self.vertices_range());
            let t = loop {
                let t = rng.gen_range(self.vertices_range());
                if t != s {
                    break t;
                }
            };

            if let Some((cut_vertices, size)) = self.min_st_vertex_cut(s, t, current_upper_bound) {
                if cut_vertices.is_empty() {
                    return None;
                }

                let smaller_partition_size =
                    size.min(self.number_of_nodes() - size - cut_vertices.len() as Node);

                if smaller_partition_size as f64 + 1.0 >= self.number_of_nodes() as f64 * imbalance
                {
                    current_upper_bound = Some(cut_vertices.len() as Node - 1);
                    current_best = Some(cut_vertices);

                    if greedy {
                        return current_best;
                    }

                    if current_upper_bound.unwrap() == 1 {
                        break;
                    }
                }
            }
        }

        current_best
    }
}

impl<T: AdjacencyList> MinVertexCut for T {}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use itertools::Itertools;
    use rand::prelude::IteratorRandom;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

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
    fn edmonds_karp() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let edges_reverse: Vec<_> = EDGES.iter().map(|(u, v)| (*v, *u)).collect();
        g.add_edges(&edges_reverse);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 3);
    }

    #[test]
    fn edmonds_karp_vertex_disjoint() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::vertex_disjoint(&g, 0, 7));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 1);
    }

    #[test]
    fn edmonds_karp_petals() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        g.add_edge(3, 7);
        g.add_edge(7, 0);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        let mf = ec.disjoint_paths();
        assert_eq!(mf.len(), 2);
    }

    #[test]
    fn edmonds_karp_no_co_arcs() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let mut ec = EdmondsKarp::new(ResidualBitMatrix::edge_disjoint(&g, 0, 7));
        let mf = ec.num_disjoint();
        assert_eq!(mf, 2);
    }

    #[test]
    fn undo_changes() {
        let mut g = AdjListMatrix::new(8);
        g.add_edges(&EDGES);
        let ec_before = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        let mut ec_after = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
        ec_after.set_remember_changes(true);
        ec_after.disjoint_paths();
        ec_after.undo_changes();

        let capacity_before = ec_before.residual_network.capacity;
        let capacity_after = ec_after.residual_network.capacity;

        for out_nodes in 0..capacity_after.len() {
            for out_node in 0..capacity_after[out_nodes].len() {
                let node_before = capacity_before[out_nodes][out_node];
                let node_after = capacity_after[out_nodes][out_node];
                assert_eq!(node_before, node_after);
            }
        }
    }

    #[test]
    fn count_num_disjoint_upto() {
        fn min(a: u32, b: u32) -> u32 {
            if a < b {
                return a;
            }
            return b;
        }
        // for node 3
        let petals_after_edge_added: [((Node, Node), u32); 13] = [
            ((0, 3), 0),
            ((3, 2), 0),
            ((2, 1), 0),
            ((1, 0), 1),
            ((3, 5), 1),
            ((5, 4), 1),
            ((4, 2), 1),
            ((4, 3), 2),
            ((4, 7), 2),
            ((7, 6), 2),
            ((6, 1), 2),
            ((6, 3), 2),
            ((3, 7), 3),
        ];
        let mut g = AdjListMatrix::new(8);
        for edge in petals_after_edge_added {
            g.add_edge(edge.0 .0, edge.0 .1);
            for k in 0..3 {
                let mut ec = EdmondsKarp::new(ResidualBitMatrix::petals(&g, 3));
                let guess = ec.count_num_disjoint_upto(k);
                let actual = min(k, edge.1);
                assert_eq!(guess, actual);
            }
        }
    }

    #[test]
    fn min_st_cut() {
        // 1 cut vertex
        {
            let graph = AdjArray::from(&[(0, 1), (2, 1)]); // 2 is not reachable from 0
            let (cut, size_s) = graph.min_st_vertex_cut(0, 2, None).unwrap();
            assert_eq!(size_s, 2); // only s is reachable
            assert_eq!(cut, vec![]);
        }

        // 1 cut vertex
        {
            let graph = AdjArray::from(&[(0, 1), (1, 2)]);
            let (cut, size_s) = graph.min_st_vertex_cut(0, 2, None).unwrap();
            assert_eq!(size_s, 1); // only s is reachable
            assert_eq!(cut, vec![1]);
        }

        // 2 cut vertices
        {
            let graph = AdjArray::from(&[(0, 1), (0, 2), (1, 3), (2, 3)]);
            let (cut, size_s) = graph.min_st_vertex_cut(0, 3, None).unwrap();
            assert_eq!(size_s, 1); // only s is reachable
            assert!(cut == vec![1, 2] || cut == vec![2, 1]);

            // upper bound
            assert!(graph.min_st_vertex_cut(0, 3, Some(1)).is_none());
        }
    }

    #[test]
    fn approx_min_cut() {
        for (n0, n1, c) in [(20, 20, 2), (20, 20, 4), (20, 20, 8)] {
            // generate two cliques with n0 and n1 nodes respectively, that are connected with a
            // 2 path. The consists of all nodes [n0+n1..n0+n1+c] if c is sufficently small

            let mut rng = Pcg64::seed_from_u64(1234);

            let mut graph = AdjArray::new((n0 + n1 + c) as usize);

            // plant cliques
            (0..n0)
                .into_iter()
                .cartesian_product((0..n0).into_iter())
                .filter(|(u, v)| u != v)
                .for_each(|(u, v)| graph.add_edge(u, v));
            (0..n1)
                .into_iter()
                .cartesian_product((0..n1).into_iter())
                .filter(|(u, v)| u != v)
                .for_each(|(u, v)| graph.add_edge(u + n0, v + n0));

            // plant connections between cliques
            for v in 0..2 * c {
                let mut us = (0..n0).into_iter().choose_multiple(&mut rng, 3);
                let mut ws = (n0..n0 + n1).into_iter().choose_multiple(&mut rng, 3);

                if v.is_even() {
                    std::mem::swap(&mut us, &mut ws);
                }

                let v = n0 + n1 + v / 2;

                us.into_iter().for_each(|u| graph.add_edge(u, v));
                ws.into_iter().for_each(|w| graph.add_edge(v, w));
            }

            let imbalance = n0.min(n1) as f64 / ((n0 + n1 + c) as f64);

            // to succeed, we need to select s from one clique and t from the other. Thus, we have
            // error rate of roughly 2^-100
            let mut cut = graph
                .approx_min_balanced_cut(&mut rng, 100, imbalance, None, false)
                .unwrap();

            assert_eq!(cut.len(), c as usize);
            cut.sort_unstable();
            assert_eq!(cut, (n0 + n1..n0 + n1 + c).into_iter().collect_vec());

            // cannot find a solution with such a large minimal component
            assert!(graph
                .approx_min_balanced_cut(&mut rng, 100, imbalance, Some(c - 1), false)
                .is_none());

            // cannot find a solution with such a large minimal component
            assert!(graph
                .approx_min_balanced_cut(&mut rng, 100, imbalance * 1.1, None, false)
                .is_none());
        }
    }
}
