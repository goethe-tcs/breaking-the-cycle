use crate::bitset::BitSet;
use crate::graph::*;
use log::info;
use num::Integer;
use std::collections::HashSet;
use std::time::Duration;

/// Using calculate_lower_bound, the main method of this file, returns both a graph and lower
/// bound of the dfvs size.
pub struct LowerBound<G> {
    graph: G,

    clique_timeout: Option<Duration>,
    max_clique: Node,
    lower_bound: Node,

    largest_clique: Option<Vec<Node>>,
    shortest_cycle: Option<Vec<Node>>,
}

impl<G> LowerBound<G>
where
    G: AdjacencyListIn
        + AdjacencyListUndir
        + GraphEdgeEditing
        + GraphOrder
        + AdjacencyTest
        + AdjacencyTestUndir
        + Clone,
{
    pub fn new(graph: &G) -> Self {
        let graph = graph.clone();

        Self {
            graph,
            clique_timeout: Some(Duration::from_millis(100)),
            max_clique: 20,
            lower_bound: 0,
            largest_clique: None,
            shortest_cycle: None,
        }
    }

    pub fn compute(&mut self) -> Node {
        if self.graph.number_of_edges() == 0 {
            return 0;
        }

        self.remove_loops();
        self.remove_cliques();
        self.remove_long_undirected_paths();
        self.unmatch();
        self.break_up_love_triangles();

        for k in 3..10 {
            self.break_up_cycles_of_length(k);
        }

        self.lower_bound
    }

    pub fn lower_bound(&self) -> Node {
        self.lower_bound
    }

    pub fn set_max_clique(&mut self, k: Node) {
        self.max_clique = k;
    }

    pub fn set_clique_search_timeout(&mut self, timeout: Option<Duration>) {
        self.clique_timeout = timeout;
    }

    pub fn largest_clique(&self) -> Option<&[Node]> {
        self.largest_clique.as_deref()
    }

    pub fn shortest_cycle(&self) -> Option<&[Node]> {
        self.shortest_cycle.as_deref()
    }

    pub fn remaining_graph(&self) -> &G {
        &self.graph
    }

    fn remove_cliques(&mut self) {
        let mut max_len = self.graph.h_index() + 1;
        if max_len == 2 {
            return;
        }

        loop {
            let clique_search = self
                .graph
                .maximum_complete_subgraph_with_timeout(Some(max_len), self.clique_timeout);

            if clique_search.1 {
                info!("Clique search timeout");
            }

            if clique_search.0.is_none() {
                break;
            }

            let clique = clique_search.0.unwrap();

            if clique.len() <= 2 {
                break;
            }

            max_len = clique.len() as Node;
            self.lower_bound += max_len - 1;

            for &u in &clique {
                self.graph.remove_edges_at_node(u);
            }

            if self.largest_clique.is_none() {
                self.largest_clique = Some(clique);
            }
        }
    }

    fn remove_loops(&mut self) {
        for u in self.graph.vertices_range() {
            if self.graph.try_remove_edge(u, u) {
                if self.shortest_cycle.is_none() {
                    self.shortest_cycle = Some(vec![u]);
                }

                self.lower_bound += 1;
            };
        }
    }

    fn unmatch(&mut self) {
        for u in self.graph.vertices_range() {
            // we avoid `if let Some(v)` to work around the borrow checker
            let v = self.graph.undir_neighbors(u).next();
            if v.is_none() {
                continue;
            }
            let v = v.unwrap();

            self.graph.remove_edges_at_node(u);
            self.graph.remove_edges_at_node(v);
            self.lower_bound += 1;

            if self.shortest_cycle.is_none() {
                self.shortest_cycle = Some(vec![u, v]);
            }
        }
    }

    fn remove_long_undirected_paths(&mut self) {
        let mut visisted = BitSet::new(self.graph.len());
        let mut stack = Vec::with_capacity(self.graph.len());
        let mut path = Vec::with_capacity(self.graph.len());
        let mut longest_path = vec![];

        loop {
            for u in self.graph.vertices() {
                if self.graph.undir_degree(u) == 0 {
                    continue;
                }

                stack.clear();
                stack.push((u, 0));
                visisted.unset_all();

                while let Some((v, len)) = stack.pop() {
                    while path.len() > len {
                        let u = path.pop().unwrap();
                        visisted.unset_bit(u as usize);
                    }

                    if visisted.set_bit(v as usize) {
                        continue;
                    }

                    path.push(v);

                    for w in self.graph.undir_neighbors(v) {
                        if !visisted[w as usize] {
                            stack.push((w, len + 1));
                        }
                    }

                    if stack.last().map_or(0, |(_, l)| *l) <= len && longest_path.len() < path.len()
                    {
                        longest_path = path.clone();
                    }
                }
            }

            if longest_path.len() < 4 {
                return;
            }

            self.lower_bound += longest_path.len() as Node / 2;
            if longest_path.len().is_odd() {
                let u = longest_path[0];
                let v = *longest_path.last().unwrap();

                if self.graph.has_edge(u, v) && self.graph.has_edge(v, u) {
                    self.lower_bound += 1;
                }
            }

            self.graph.remove_edges_of_nodes(&longest_path);
            longest_path.clear();
        }
    }

    fn break_up_love_triangles(&mut self) {
        // for all edges going from u to v, we look whether any of v's neighbors has an edge to u
        for u in self.graph.vertices_range() {
            if self.graph.in_degree(u) == 0 || self.graph.out_degree(u) == 0 {
                continue;
            };

            let in_neigh: HashSet<Node> = self.graph.in_neighbors(u).into_iter().collect();

            let vw = (|| {
                for v in self.graph.out_neighbors(u) {
                    if let Some(w) = self.graph.out_neighbors(v).find(|w| in_neigh.contains(w)) {
                        return Some((v, w));
                    }
                }
                None
            })();

            if let Some((v, w)) = vw {
                self.graph.remove_edges_at_node(u);
                self.graph.remove_edges_at_node(v);
                self.graph.remove_edges_at_node(w);
                self.lower_bound += 1;

                if self.shortest_cycle.is_none() {
                    self.shortest_cycle = Some(vec![u, v, w]);
                }
            }
        }
    }

    fn break_up_cycles_of_length(&mut self, k: Node) {
        let mut parent = vec![0; self.graph.len()];
        let mut cycle = Vec::with_capacity(k as usize - 1);

        // for all edges going from u to v, we look whether any of v's neighbors has an edge to u
        for u in self.graph.vertices_range() {
            if self.graph.in_degree(u) == 0 || self.graph.out_degree(u) == 0 {
                continue;
            };

            // compute parent for all nodes on a cycle of length k including node u
            {
                parent[u as usize] = self.graph.number_of_nodes();
                let mut bfs = self.graph.bfs_with_predecessor(u);
                bfs.next();
                bfs.include_node(u);
                bfs.stop_at(u);
                bfs.parent_array_into(&mut parent);

                if parent[u as usize] == self.graph.number_of_nodes() {
                    continue; // not on cycle
                }
            }

            let mut node = parent[u as usize];
            cycle.clear();
            while node != u && cycle.len() + 1 < k as usize {
                cycle.push(node);
                node = parent[node as usize];
            }

            if node != u || cycle.len() + 1 != k as usize {
                continue; // cycle too long
            }

            for &v in &cycle {
                self.graph.remove_edges_at_node(v);
            }
            self.graph.remove_edges_at_node(u);
            self.lower_bound += 1;

            if self.shortest_cycle.is_none() {
                let mut cycle = cycle.clone();
                cycle.push(u);
                self.shortest_cycle = Some(cycle);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;

    // unit tests for search_and_destroy at k=2 aka unmatch and the call and return of unmatch
    fn create_test_graph1() -> AdjArrayIn {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (0, 2), (3, 3)]);
        graph
    }

    fn create_test_graph2() -> AdjListMatrixIn {
        let graph = AdjListMatrixIn::from(&[(0, 1), (1, 0), (1, 2), (2, 1)]);
        graph
    }

    fn create_test_graph3() -> AdjListMatrixIn {
        let graph = AdjListMatrixIn::from(&[
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (3, 4),
            (4, 3),
        ]);
        graph
    }

    #[test]
    fn test_1() {
        let mut source_graph = create_test_graph1();
        source_graph.remove_edge(3, 3);
        let graph = create_test_graph1();
        let mut lb = LowerBound::new(&graph);
        lb.set_clique_search_timeout(None);
        lb.compute();

        assert_eq!(source_graph.edges_vec(), lb.remaining_graph().edges_vec());
        assert_eq!(lb.lower_bound(), 1);
    }

    #[test]
    fn test_2() {
        let graph = create_test_graph2();
        let mut lb = LowerBound::new(&graph);
        lb.set_clique_search_timeout(None);
        lb.compute();

        assert_eq!(lb.remaining_graph().number_of_edges(), 0);
        assert_eq!(lb.lower_bound(), 1);
    }

    #[test]
    fn test_3() {
        let mut graph = create_test_graph1();
        graph.remove_edge(0, 2);
        graph.add_edge(2, 0);

        let mut lb = LowerBound::new(&graph);
        lb.set_clique_search_timeout(None);
        lb.compute();

        assert_eq!(lb.remaining_graph().number_of_edges(), 0);
        assert_eq!(lb.lower_bound(), 2);
    }

    #[test]
    fn test_4() {
        let mut graph = create_test_graph3();
        graph.remove_edge(3, 4);
        graph.remove_edge(4, 3);
        let mut lb = LowerBound::new(&graph);
        lb.set_clique_search_timeout(None);
        lb.remove_cliques();
        assert_eq!(lb.remaining_graph().number_of_edges(), 0);
        assert_eq!(lb.lower_bound(), 2);
    }

    #[test]
    fn test_5() {
        let graph = create_test_graph3();
        let mut lb = LowerBound::new(&graph);
        lb.set_clique_search_timeout(None);
        lb.compute();
        assert_eq!(lb.remaining_graph().number_of_edges(), 0);
        assert_eq!(lb.lower_bound(), 3);
    }

    #[test]
    fn break_up_cycles_of_length() {
        let maxk: Node = 6;
        let mut graph = AdjArrayIn::new((maxk * (maxk + 1) / 2) as usize);

        let mut start = 0;
        for i in 1..=maxk {
            graph.connect_cycle((start..start + i).into_iter());
            start += i;
        }

        for i in 2..=maxk {
            let mut lb = LowerBound::new(&graph);
            lb.set_clique_search_timeout(None);
            lb.break_up_cycles_of_length(i);
            assert_eq!(lb.lower_bound(), 1);
        }
    }
}
