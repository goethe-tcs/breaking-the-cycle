use super::*;
use fxhash::FxHashMap;
use itertools::Itertools;
use std::cmp::Ordering;
use std::fmt;

pub trait Setter: Sized {
    /// Creates a mapper where the largest node that can be inserted is n-1.
    fn with_capacity(n: Node) -> Self;

    /// Creates a mapper where the largest node that can be handled is n-1.
    /// Each mapping is of form x <-> x for all x.
    /// Subsequent calls to [`Setter::map_node_to`] are forbidden.
    fn identity(n: Node) -> Self;

    /// Stores a mapping old <-> new
    fn map_node_to(&mut self, old: Node, new: Node);

    /// Constructs a mapper from a Node-slice `new_ids` where `new_ids[i]` stores the new id of the
    /// old node `i`.
    ///
    /// # Example
    ///
    /// ```
    /// use dfvs::graph::{NodeMapper, Setter, Getter};
    /// let mapper = NodeMapper::from_rank(&[2, 0, 1]);
    /// assert_eq!(mapper.new_id_of(0), Some(2));
    /// assert_eq!(mapper.new_id_of(1), Some(0));
    /// assert_eq!(mapper.new_id_of(2), Some(1));
    /// ```
    fn from_rank(new_ids: &[Node]) -> Self {
        let cap = new_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(new_ids.len() as Node);
        let mut res = Self::with_capacity(cap + 1);
        for (old, &new) in new_ids.iter().enumerate() {
            res.map_node_to(old as Node, new);
        }
        res
    }

    /// Constructs a mapper from a sequence of tuples (old, new).
    ///
    /// # Example
    ///
    /// ```
    /// use dfvs::graph::{NodeMapper, Setter, Getter};
    /// let mapper = NodeMapper::from_sequence(&[(0,2), (2, 1)]);
    /// assert_eq!(mapper.new_id_of(0), Some(2));
    /// assert_eq!(mapper.new_id_of(1), None);
    /// assert_eq!(mapper.new_id_of(2), Some(1));
    /// ```
    fn from_sequence(seq: &[(Node, Node)]) -> Self {
        if seq.is_empty() {
            return Self::with_capacity(0);
        }

        let cap = seq.iter().map(|&(o, n)| o.max(n)).max().unwrap();

        let mut res = Self::with_capacity(cap + 1);
        for &(old, new) in seq {
            res.map_node_to(old as Node, new);
        }
        res
    }
}

pub trait Getter {
    /// If the mapping (old, new) exists, returns Some(new), otherwise None
    fn new_id_of(&self, old: Node) -> Option<Node>;

    /// If the mapping (old, new) exists, returns Some(old), otherwise None
    fn old_id_of(&self, new: Node) -> Option<Node>;

    /// Returns the number of explicitly stored mappings; 0 for identity
    fn len(&self) -> Node;

    /// Returns true if no mapping is stored; true for identity
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Applies [`Getter::old_id_of`] to each iterator item. Uses the iterator item (new) as a fallback
    /// if the mapping(old, new) doesn't exist.
    fn get_old_ids(&self, new_ids: impl Iterator<Item = Node>) -> Vec<Node> {
        new_ids
            .map(|new| self.old_id_of(new).unwrap_or(new))
            .collect_vec()
    }

    /// Applies [`Getter::new_id_of`] to each iterator item. Uses the iterator item (old) as a fallback
    /// if the mapping(old, new) doesn't exist.
    fn get_new_ids(&self, old_ids: impl Iterator<Item = Node>) -> Vec<Node> {
        old_ids
            .map(|old| self.new_id_of(old).unwrap_or(old))
            .collect_vec()
    }

    /// Create a 'copy' of type `GO` from the input graph where all nodes are relabelled according to this mapper.
    /// Any node u (and its incident edges) is dropped if there [`Getter::new_id_of`] returns None.
    ///
    /// # Example
    ///
    /// ```
    /// use dfvs::graph::*;
    /// let input = AdjArray::from(&[(0,1), (1,2), (2, 2)]);
    /// let mut map = NodeMapper::with_capacity(3);
    /// // node 0 gets dropped, 1 becomes 3 and 2 becomes 1
    /// map.map_node_to(1, 3);
    /// map.map_node_to(2, 1);
    /// let output : AdjArray = map.relabelled_graph_as(&input);
    /// assert_eq!(output.edges_vec(), vec![(1, 1), (3, 1)]);
    /// ```
    fn relabelled_graph_as<GI, GO>(&self, input: &GI) -> GO
    where
        GI: GraphOrder + AdjacencyList,
        GO: GraphNew + GraphEdgeEditing,
    {
        let max_node = input
            .vertices()
            .flat_map(|u| self.new_id_of(u).map(|x| x + 1))
            .max()
            .unwrap_or(0);

        if max_node == 0 {
            return GO::new(0);
        }

        let mut output = GO::new(max_node as usize);
        for old_u in input.vertices() {
            if let Some(new_u) = self.new_id_of(old_u) {
                for new_v in input.out_neighbors(old_u).flat_map(|u| self.new_id_of(u)) {
                    output.try_add_edge(new_u, new_v);
                }
            }
        }

        output
    }

    /// Short-hand for [`Getter::relabelled_graph_as`] where the output type matches the input type.
    fn relabelled_graph<G>(&self, input: &G) -> G
    where
        G: AdjacencyList + GraphNew + GraphEdgeEditing,
    {
        self.relabelled_graph_as::<G, G>(input)
    }
}

pub trait Compose {
    /// Takes two Mapper M1 (original -> intermediate) and M2 (intermediate -> final)
    /// and produces a new mapper (original -> final). All mappings without a correspondence
    /// in the other mapper are dropped
    fn compose(first: &Self, second: &Self) -> Self;
}

pub trait Inverse {
    /// Returns a new mapper where for each mapping (a, b) of the original, there exists
    /// a mapping (b, a) in the new mapper
    #[must_use]
    fn inverse(&self) -> Self;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This Node Mapper cannot be read from, and all insertions are dumped.
/// It can be used to optimize a way the cost of producing a mapping if it is not used
pub struct WriteOnlyNodeMapper {}

impl Setter for WriteOnlyNodeMapper {
    fn with_capacity(_: Node) -> Self {
        Self {}
    }

    fn identity(_n: Node) -> Self {
        Self {}
    }

    fn map_node_to(&mut self, _old: Node, _new: Node) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone)]
pub struct NodeMapper {
    new_to_old: FxHashMap<Node, Node>,
    old_to_new: FxHashMap<Node, Node>,
    is_identity: bool,
}

impl Setter for NodeMapper {
    fn with_capacity(n: Node) -> Self {
        Self {
            new_to_old: FxHashMap::with_capacity_and_hasher(n as usize, Default::default()),
            old_to_new: FxHashMap::with_capacity_and_hasher(n as usize, Default::default()),
            is_identity: false,
        }
    }

    fn identity(_n: Node) -> Self {
        let mut res = Self::with_capacity(0);
        res.is_identity = true;
        res
    }

    fn map_node_to(&mut self, old: Node, new: Node) {
        assert!(!self.is_identity);
        let success =
            self.old_to_new.insert(old, new).is_none() & self.new_to_old.insert(new, old).is_none();
        assert!(success);
    }
}

impl Getter for NodeMapper {
    fn new_id_of(&self, old: Node) -> Option<Node> {
        if self.is_identity {
            Some(old)
        } else {
            Some(*self.old_to_new.get(&old)?)
        }
    }

    fn old_id_of(&self, new: Node) -> Option<Node> {
        if self.is_identity {
            Some(new)
        } else {
            Some(*self.new_to_old.get(&new)?)
        }
    }

    fn len(&self) -> Node {
        self.old_to_new.len() as Node
    }
}

impl Compose for NodeMapper {
    fn compose(first: &NodeMapper, second: &NodeMapper) -> Self {
        if first.is_identity {
            return second.clone();
        }

        if second.is_identity {
            return first.clone();
        }

        let mut composition = Self::with_capacity(second.len() as Node);
        for (&original, &intermediate) in first.old_to_new.iter() {
            if let Some(new) = second.new_id_of(intermediate) {
                composition.map_node_to(original, new);
            }
        }
        composition
    }
}

impl Inverse for NodeMapper {
    fn inverse(&self) -> Self {
        Self {
            old_to_new: self.new_to_old.clone(),
            new_to_old: self.old_to_new.clone(),
            is_identity: self.is_identity,
        }
    }
}

impl fmt::Debug for NodeMapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            format!(
                "[{}]",
                self.old_to_new
                    .iter()
                    .map(|(&o, &n)| format!("{}<->{}", o, n))
                    .join(", ")
            )
            .as_str(),
        )?;

        Ok(())
    }
}

/// The [`RankingForwardMapper`] is less powerful than [`NodeMapper`] (e.g. it is lacking a
/// [`Setter`] interface and cannot deal with missing mappings) but is much more efficient for
/// the cases where it is applicable.
#[derive(Clone)]
pub struct RankingForwardMapper {
    new_ids: Vec<Node>,
}

impl RankingForwardMapper {
    /// Functionally equivalent to [`Setter::from_rank`], but takes and consume a vector.
    /// In other words, a [`RankingForwardMapper`] is constructed from the vector `new_ids`,
    /// where `new_ids[i]` stores the new id of the old node `i`.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::node_mapper::{RankingForwardMapper, Getter};
    /// let mapper = RankingForwardMapper::from_vec(vec![4,0,1]);
    /// assert_eq!(mapper.new_id_of(0), Some(4));
    /// assert_eq!(mapper.new_id_of(1), Some(0));
    /// assert_eq!(mapper.new_id_of(2), Some(1));
    /// ```
    pub fn from_vec(new_ids: Vec<Node>) -> Self {
        Self { new_ids }
    }

    /// Ranks all nodes in `graph` according to the provided comparator `compare` and constructs
    /// a RankingForwardMapper from it.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::{AdjacencyList, AdjArray};
    /// use dfvs::graph::node_mapper::{RankingForwardMapper, Getter};
    /// let graph = AdjArray::from(&[(0, 1), (0,2), (0,3), (1,2), (1,3), (2,3)]);
    /// let mapper = RankingForwardMapper::from_graph_sort_by(&graph, |a, b|
    ///   graph.out_degree(a).cmp(&graph.out_degree(b))
    /// );
    /// assert_eq!(mapper.new_id_of(3), Some(0));
    /// assert_eq!(mapper.new_id_of(2), Some(1));
    /// assert_eq!(mapper.new_id_of(1), Some(2));
    /// assert_eq!(mapper.new_id_of(0), Some(3));
    /// ```
    pub fn from_graph_sort_by<G, F>(graph: &G, mut compare: F) -> Self
    where
        G: GraphOrder,
        F: FnMut(Node, Node) -> Ordering,
    {
        let mut vertices: Vec<Node> = graph.vertices_range().collect();
        vertices.sort_by(|&a, &b| compare(a, b));
        Self::from_vec(vertices)
    }

    /// Ranks all nodes in `graph` according to the provided `key` function increasingly and constructs
    /// a RankingForwardMapper from it.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::{AdjacencyList, AdjArray};
    /// use dfvs::graph::node_mapper::{RankingForwardMapper, Getter};
    /// let graph = AdjArray::from(&[(0, 1), (0,2), (0,3), (1,2), (1,3), (2,3)]);
    /// let mapper = RankingForwardMapper::from_graph_sort_by_key(&graph, |u| graph.out_degree(u));
    /// assert_eq!(mapper.new_id_of(3), Some(0));
    /// assert_eq!(mapper.new_id_of(2), Some(1));
    /// assert_eq!(mapper.new_id_of(1), Some(2));
    /// assert_eq!(mapper.new_id_of(0), Some(3));
    /// ```
    pub fn from_graph_sort_by_key<G, F, K>(graph: &G, mut key: F) -> Self
    where
        G: GraphOrder,
        F: FnMut(Node) -> K,
        K: Ord,
    {
        Self::from_graph_sort_by(graph, |a, b| key(a).cmp(&key(b)))
    }

    /// Ranks all nodes in `graph` according to the provided `key` function increasingly and constructs
    /// a RankingForwardMapper from it.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::{AdjacencyList, AdjArray};
    /// use dfvs::graph::node_mapper::{RankingForwardMapper, Getter};
    /// let graph = AdjArray::from(&[(0, 1), (0,2), (0,3), (1,2), (1,3), (2,3)]);
    /// let mapper = RankingForwardMapper::from_graph_sort_by_key_reverse(&graph, |u| graph.out_degree(u));
    /// assert_eq!(mapper.new_id_of(3), Some(3));
    /// assert_eq!(mapper.new_id_of(2), Some(2));
    /// assert_eq!(mapper.new_id_of(1), Some(1));
    /// assert_eq!(mapper.new_id_of(0), Some(0));
    /// ```
    pub fn from_graph_sort_by_key_reverse<G, F, K>(graph: &G, mut key: F) -> Self
    where
        G: GraphOrder,
        F: FnMut(Node) -> K,
        K: Ord,
    {
        Self::from_graph_sort_by(graph, |a, b| key(b).cmp(&key(a)))
    }
}

impl Getter for RankingForwardMapper {
    fn new_id_of(&self, old: Node) -> Option<Node> {
        if (old as usize) < self.new_ids.len() {
            Some(self.new_ids[old as usize])
        } else {
            None
        }
    }

    /// If the mapping (old, new) exists, returns Some(old), otherwise None.
    ///
    /// # Warning
    /// This implementation is provided for compatibility reasons but takes linear time.
    /// If this method is required, consider using a `NodeMapper` instance instead.
    fn old_id_of(&self, new: Node) -> Option<Node> {
        self.new_ids
            .iter()
            .find_position(|&&x| x == new)
            .map(|(i, _)| i as Node)
    }

    fn len(&self) -> Node {
        self.new_ids.len() as Node
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_node_mapper() {
        let mut m1 = NodeMapper::with_capacity(10);
        assert!(m1.is_empty());
        for x in 0..5 {
            m1.map_node_to(2 * x, x);
            assert!(!m1.is_empty());
            assert_eq!(m1.len(), x + 1);
        }

        let mut m2 = NodeMapper::with_capacity(5);
        m2.map_node_to(0, 3);
        m2.map_node_to(1, 2);
        m2.map_node_to(2, 1);

        let m = NodeMapper::compose(&m1, &m2);

        let result: Vec<Option<Node>> = (0..10).map(|x| m.new_id_of(x)).collect();
        assert_eq!(
            result,
            [
                Some(3),
                None,
                Some(2),
                None,
                Some(1),
                None,
                None,
                None,
                None,
                None
            ]
        );
    }

    #[test]
    fn test_compose() {
        let mappings = vec![(0, 3), (10, 2), (20, 1)];

        let mut map = NodeMapper::with_capacity(5);
        for &(u, v) in &mappings {
            map.map_node_to(u, v);
        }

        let id = NodeMapper::identity(50);

        {
            let comp = NodeMapper::compose(&id, &map);
            for &(u, v) in &mappings {
                assert_eq!(comp.new_id_of(u), Some(v));
                assert_eq!(comp.old_id_of(v), Some(u));
            }
        }

        {
            let comp = NodeMapper::compose(&map, &id);
            for &(u, v) in &mappings {
                assert_eq!(comp.new_id_of(u), Some(v));
                assert_eq!(comp.old_id_of(v), Some(u));
            }
        }
    }

    #[test]
    fn test_node_mapper_inverse() {
        let mut m2 = NodeMapper::with_capacity(5);
        assert!(m2.is_empty());
        m2.map_node_to(0, 3);
        m2.map_node_to(10, 2);
        m2.map_node_to(20, 1);
        assert!(!m2.is_empty());

        let inv = m2.inverse();
        let result: Vec<Option<Node>> = (0..5).map(|x| inv.new_id_of(x)).collect();
        assert_eq!(result, vec![None, Some(20), Some(10), Some(0), None]);

        for i in 0..20 {
            assert_eq!(inv.new_id_of(i), m2.old_id_of(i));
            assert_eq!(m2.new_id_of(i), inv.old_id_of(i));
        }
    }

    #[test]
    fn test_identity() {
        let map = NodeMapper::identity(123);
        assert_eq!(map.old_id_of(12), Some(12));
        assert_eq!(map.new_id_of(5), Some(5));
    }

    #[test]
    fn test_format() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(2, 3);
        map.map_node_to(5, 4);
        let text = format!("{:?}", map);
        assert!(text.contains("2<->3"));
        assert!(text.contains("5<->4"));
    }

    #[test]
    #[should_panic]
    fn test_collision_on_old() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(2, 3);
        map.map_node_to(2, 4);
    }

    #[test]
    #[should_panic]
    fn test_collision_on_new() {
        let mut map = NodeMapper::with_capacity(10);
        map.map_node_to(1, 3);
        map.map_node_to(2, 3);
    }

    #[test]
    fn relabelling() {
        // keep all
        {
            let graph = AdjArray::from(&[(0, 1), (1, 2)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(0, 2);
            map.map_node_to(1, 1);
            map.map_node_to(2, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(graph.edges_vec(), vec![(1, 0), (2, 1)]);
            assert_eq!(graph.len(), 3);
        }

        // drop 1
        {
            let graph = AdjArray::from(&[(0, 1), (1, 2)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(0, 1);
            map.map_node_to(2, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(graph.edges_vec(), vec![]);
            assert_eq!(graph.len(), 2);
        }

        // loop at node 0 (previous bug)
        {
            let graph = AdjArray::from(&[(1, 1)]);
            let mut map = NodeMapper::with_capacity(10);
            map.map_node_to(1, 0);
            let graph = map.relabelled_graph(&graph);

            assert_eq!(graph.edges_vec(), vec![(0, 0)]);
            assert_eq!(graph.len(), 1);
        }
    }
}
