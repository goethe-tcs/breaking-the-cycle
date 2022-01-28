use super::*;
use crate::bitset::BitSet;
use itertools::Itertools;
use rand::Rng;

pub trait GeneratorSubstructures {
    // connects the nodes passed with a simple path
    fn connect_path<T: IntoIterator<Item = Node>>(&mut self, nodes_on_path: T);

    // connects the nodes passed with a simple path and adds a connection between the first and the last node
    fn connect_cycle<T: IntoIterator<Item = Node>>(&mut self, nodes_in_cycle: T);

    // generates a clique of all nodes includes in the bit
    fn connect_nodes(&mut self, nodes: &BitSet, with_loops: bool);
}

impl<G: GraphEdgeEditing> GeneratorSubstructures for G {
    fn connect_path<T: IntoIterator<Item = Node>>(&mut self, nodes_on_path: T) {
        for (u, v) in nodes_on_path.into_iter().tuple_windows() {
            self.add_edge(u, v);
        }
    }

    fn connect_cycle<T: IntoIterator<Item = Node>>(&mut self, nodes_in_cycle: T) {
        let mut iter = nodes_in_cycle.into_iter();

        // we use a rather tedious implementation to avoid needing to clone the iterator
        if let Some(first) = iter.next() {
            let mut prev = first;
            for cur in iter {
                self.add_edge(prev, cur);
                prev = cur;
            }

            self.add_edge(prev, first);
        }
    }

    fn connect_nodes(&mut self, nodes: &BitSet, with_loops: bool) {
        for u in nodes.iter() {
            for v in nodes.iter() {
                if !with_loops && u == v {
                    continue;
                }

                self.add_edge(u as Node, v as Node);
            }
        }
    }
}

/// Generates a random spanning tree over a presumed complete graph
/// where all edges are orientated away from the root
/// by carrying out a naive loop-less random walk
pub fn random_mst<G: GraphNew + GraphEdgeEditing, R: Rng>(rng: &mut R, n: usize) -> (G, u32) {
    if n == 1 {
        return (G::new(1), 0);
    }
    assert!(n > 1);

    let mut mst = G::new(n);
    let mut connected = BitSet::new(n);

    let mut get_random_node = || rng.gen_range(0_usize..n);

    let root = get_random_node();
    connected.set_bit(root);

    let mut path = Vec::new();
    let mut on_path = BitSet::new(n);

    while !connected.full() {
        on_path.unset_all();
        path.clear();

        loop {
            // sample random node not on path yet
            let u = get_random_node();

            if path.is_empty() && connected[u] {
                // the first not has to be unconnected; reject otherwise
                // TODO: this rejection step can be a bottleneck for larger graphs
                // we may want to a hash set containing unconnected graphs
                continue;
            }

            if on_path.set_bit(u) {
                // avoid loops!
                continue;
            }
            path.push(u);

            if connected.set_bit(u) {
                // our path reached the tree -- so connect it and stop
                for edge in path.windows(2) {
                    mst.add_edge(edge[1] as u32, edge[0] as u32);
                }

                break;
            }
        }
    }

    (mst, root as Node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect_path() {
        {
            let mut g = AdjListMatrix::new(6);
            g.connect_path([]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_path([1]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_path([2, 1]);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(2, 1));
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_path([0, 3, 1, 4]);
            let mut edges = g.edges();
            edges.sort();
            assert_eq!(g.edges(), vec![(0, 3), (1, 4), (3, 1)]);
        }
    }

    #[test]
    fn test_connect_cycle() {
        {
            let mut g = AdjListMatrix::new(6);
            g.connect_cycle([]);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_cycle([1]);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(1, 1));
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_cycle([0, 3, 1, 4]);
            let mut edges = g.edges();
            edges.sort();
            assert_eq!(g.edges(), vec![(0, 3), (1, 4), (3, 1), (4, 0)]);
        }
    }

    #[test]
    fn test_connect_nodes() {
        {
            let mut g = AdjListMatrix::new(6);
            g.connect_nodes(&BitSet::new(6), true);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_nodes(&BitSet::from_slice(6, &[1]), false);
            assert_eq!(g.number_of_edges(), 0);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_nodes(&BitSet::from_slice(6, &[1]), true);
            assert_eq!(g.number_of_edges(), 1);
            assert!(g.has_edge(1, 1));
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_nodes(&BitSet::from_slice(6, &[1, 2, 4]), false);
            assert_eq!(g.number_of_edges(), 6);
        }

        {
            let mut g = AdjListMatrix::new(6);
            g.connect_nodes(&BitSet::from_slice(6, &[1, 2, 4]), true);
            assert_eq!(g.number_of_edges(), 9);
        }
    }

    #[test]
    fn test_mst() {
        for n in 2_usize..50_usize {
            let res = random_mst(&mut rand::thread_rng(), n);
            let mst: AdjListMatrix = res.0;
            let root = res.1;

            assert_eq!(mst.number_of_nodes() as usize, n);
            assert_eq!(mst.number_of_edges(), n - 1);
            assert!(root < mst.number_of_nodes());
            assert_eq!(mst.dfs(root).count(), n); // check orientation of edges away from root
        }
    }
}
