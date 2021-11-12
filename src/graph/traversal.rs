use crate::bitset::BitSet;
use std::collections::VecDeque;
use super::*;

pub struct TraversalState<'a, G> {
    graph: &'a G,
    visited: BitSet
}

impl<'a, G : AdjecencyList> TraversalState<'a, G> {
    pub fn new(graph: &'a G) -> TraversalState<'a, G> {
        Self {graph, visited: BitSet::new(graph.len())}
    }

    /// Executes breadth-first-search starting at node `start` and calls
    /// callback for every node visited in order
    pub fn bfs_directed<T: FnMut(Node)>(&mut self, start: Node, mut callback: T) {
        let mut queue: VecDeque<Node> = VecDeque::from(vec![start]);
        while let Some(u) = queue.pop_front() {
            callback(u);
            for &v in self.graph.out_neighbors(u) {
                if !self.visited.set_bit(v as usize) {
                    queue.push_back(v);
                }
            };
        }
    }

    /// Executes breadth-first-search starting at node `start` and calls
    /// callback for every node visited in order
    pub fn bfs_undirected<T: FnMut(Node)>(&mut self, start: Node, mut callback: T) {
        let mut queue: VecDeque<Node> = VecDeque::from(vec![start]);
        while let Some(u) = queue.pop_front() {
            callback(u);
            for &v in self.graph.out_neighbors(u).iter().chain(self.graph.in_neighbors(u).iter()) {
                if !self.visited.set_bit(v as usize) {
                    queue.push_back(v);
                }
            };
        }
    }

    /// Executes depth-first-search starting at node `start` and calls
    /// callback for every node visited in order
    pub fn dfs_directed<T: FnMut(Node)>(&mut self, start: Node, mut callback: T) {
        let mut stack: Vec<Node> = vec![start];
        while let Some(u) = stack.pop() {
            callback(u);
            for &v in self.graph.out_neighbors(u) {
                if !self.visited.set_bit(v as usize) {
                    stack.push(v);
                }
            };
        }
    }

    /// Executes depth-first-search starting at node `start` and calls
    /// callback for every node visited in order
    pub fn dfs_undirected<T: FnMut(Node)>(&mut self, start: Node, mut callback: T) {
        let mut stack: Vec<Node> = vec![start];
        while let Some(u) = stack.pop() {
            callback(u);
            for &v in self.graph.out_neighbors(u).iter().chain(self.graph.in_neighbors(u).iter()) {
                if !self.visited.set_bit(v as usize) {
                    stack.push(v);
                }
            };
        }
    }

    pub fn visited(&self) -> &BitSet {
        &self.visited
    }

    pub fn did_visit(&self, u : Node) -> bool {
        self.visited[u as usize]
    }
}

pub enum TravAlgo {
    DirBFS(Node), UndirBFS(Node),
    DirDFS(Node), UndirDFS(Node),
    TopoSearch
}

pub trait Traversal : AdjecencyList + Sized {
    /// Visits as many nodes as possible without violating the topological order,
    /// i.e. before a node is visited, all its predecessors are visited. If and only if
    /// the graph is acyclic all nodes are visited.
    fn topology_search<T: FnMut(Node)>(&self, mut callback : T) {
        let mut in_degs : Vec<Node> = self.vertices().map(|u| {self.in_degree(u)}).collect();

        let mut stack : Vec<Node> = in_degs.iter().enumerate()
            .filter_map(|(u, &d)| {if d == 0 {Some(u as Node)} else {None}})
            .collect();

        while let Some(u) = stack.pop() {
            callback(u);

            for &v in self.out_neighbors(u) {
                in_degs[v as usize] -= 1;
                if in_degs[v as usize] == 0 {
                    stack.push(v);
                }
            }
        }
    }

    /// Runs the requested algorithm and invokes the callback for every visited node
    fn traverse<T: FnMut(Node)>(&self, algo: TravAlgo, callback : T) {
        if let TravAlgo::TopoSearch = algo {
            return self.topology_search(callback);
        }

        let mut state = TraversalState::new(self);

        match algo {
            TravAlgo::DirBFS(start)   => {state.bfs_directed  (start, callback);},
            TravAlgo::UndirBFS(start) => {state.bfs_undirected(start, callback);},
            TravAlgo::DirDFS(start)   => {state.dfs_directed  (start, callback);},
            TravAlgo::UndirDFS(start) => {state.dfs_undirected(start, callback);},
            TravAlgo::TopoSearch => {/* unreachable */}
        }
    }


    /// Runs the requested algorithm and returns the order in which nodes where visited
    fn traversal_order(&self, algo: TravAlgo) -> Vec<Node> {
        let mut order = Vec::with_capacity(self.len());
        self.traverse(algo, |u| { order.push(u); });
        order
    }

    /// Runs the requested algorithm and returns ranks of nodes, i.e.
    /// if the i-th visited node was u, then rank[i] = u. Returns
    /// a ranking if all nodes were visited exactly one, otherwise None
    fn traversal_rank(&self, algo: TravAlgo) -> Option<Vec<Node>> {
        let mut ranks = vec![0; self.len()];
        let mut rank : Node = 0;

        self.traverse(algo,  |u| {
            assert_eq!(ranks[u as usize], 0);
            ranks[u as usize] = rank;
            rank += 1;
        });

        if rank == self.number_of_nodes() {Some(ranks)} else {None}
    }
}

impl<G : AdjecencyList> Traversal for G {}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn bfs_order() {
        let mut graph = AdjListMatrix::new(6);
        //  / 2 --- \
        // 1         4 - 3
        //  \ 0 - 5 /
        for (u, v) in [(1,2), (1,0), (4,3), (0,5), (2,4), (5, 4)] {
            graph.add_edge(u, v);
        }

        {
            let order = graph.traversal_order(TravAlgo::DirBFS(1));
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);
            assert!((order[1] == 0 && order[2] == 2) || (order[2] == 0 && order[1] == 2));
            assert!((order[3] == 4 && order[4] == 5) || (order[4] == 4 && order[3] == 5));
            assert_eq!(order[5], 3);
        }

        {
            let order = graph.traversal_order(TravAlgo::DirBFS(5));
            assert_eq!(order, [5,4,3]);
        }
    }

    #[test]
    fn dfs_order() {
        let mut graph = AdjListMatrix::new(6);
        //  / 2
        // 1         4 - 3
        //  \ 0 - 5 /
        for (u, v) in [(1,2), (1,0), (4,3), (0,5), (5, 4)] {
            graph.add_edge(u, v);
        }

        {
            let order = graph.traversal_order(TravAlgo::DirDFS(1));
            assert_eq!(order.len(), 6);

            assert_eq!(order[0], 1);

            if order[1] == 2 {
                assert_eq!(order[2..6], [0, 5, 4, 3]);
            } else {
                assert_eq!(order[1..6], [0, 5, 4, 3, 2]);
            }
        }

        {
            let order = graph.traversal_order(TravAlgo::DirDFS(5));
            assert_eq!(order, [5,4,3]);
        }
    }

    #[test]
    fn topology_rank() {
        let mut graph = AdjListMatrix::new(8);
        graph.add_edges(&[(2,0), (1,0), (0,3), (0,4), (0,5), (3,6)]);

        {
            let ranks = graph.traversal_rank(TravAlgo::TopoSearch).unwrap();
            assert_eq!(*ranks.iter().min().unwrap(), 0);
            assert_eq!(*ranks.iter().max().unwrap(), graph.number_of_nodes() - 1);
            for (u, v) in graph.edges() {
                assert!(ranks[u as usize] < ranks[v as usize]);
            }
        }

        graph.add_edge(6, 2); // introduce cycle
        {
            let topo = graph.traversal_rank(TravAlgo::TopoSearch);
            assert!(topo.is_none());
        }
    }
}
