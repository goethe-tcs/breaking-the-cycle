use super::*;
use itertools::Itertools;

pub type PartitionClass = Node;

/// A partition splits a graph into node-disjoint substructures (think SCCs, bipartite classes, etc)
pub struct Partition {
    // Remark on the encoding: in a perfect world `classes` should contain `Option<PartitionClass>`
    // to encode "unassigned" nodes. As of writing, this is extremely wasteful since `PartitionClass`
    // requires 4 bytes, while `Option<PartitionClass>` takes 8 bytes (due to padding for alignment).
    // We hence treat class 0 as unassigned and hide that from the user. Partition class `i` is
    // then mapped to the internal class `i+1`; we use this mapping (e.g. instead of encoding
    // unassigned with MAXINT) to simplify the interplay between `classes` and `class_sizes`.
    classes: Vec<PartitionClass>,
    class_sizes: Vec<Node>,
}

impl Partition {
    /// Creates a partition for `nodes` nodes which are initially all unassigned
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let partition = Partition::new(10);
    /// assert_eq!(partition.number_of_unassigned(), 10);
    /// ```
    pub fn new(nodes: Node) -> Self {
        Self {
            classes: vec![0; nodes as usize],
            class_sizes: vec![nodes],
        }
    }

    /// Creates a new partition class and assigns all provided nodes to it; we require that these
    /// nodes were previously unassigned.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_unassigned(), 8);
    /// assert_eq!(partition.number_in_class(class_id), 2);
    /// assert_eq!(partition.class_of_edge(1, 2), None);
    /// assert_eq!(partition.class_of_edge(2, 4), Some(class_id));
    /// ```
    pub fn add_class<I: IntoIterator<Item = Node>>(&mut self, nodes: I) -> PartitionClass {
        let class_id = self.class_sizes.len() as PartitionClass;
        self.class_sizes.push(0);

        let size = &mut self.class_sizes[class_id as usize];
        for u in nodes {
            assert_eq!(self.classes[u as usize], 0); // check that node is unassigned
            self.classes[u as usize] = class_id;
            *size += 1;
        }

        self.class_sizes[0] -= *size;

        class_id - 1
    }

    /// Moves node into an existing partition class. The node may or may not have been previously assigned.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// partition.move_node(1, class_id);
    /// assert_eq!(partition.number_of_unassigned(), 7);
    /// assert_eq!(partition.number_in_class(class_id), 3);
    /// assert_eq!(partition.class_of_edge(1, 2), Some(class_id));
    /// ```
    pub fn move_node(&mut self, node: Node, new_class: PartitionClass) {
        self.class_sizes[self.classes[node as usize] as usize] -= 1;
        self.classes[node as usize] = new_class + 1;
        self.class_sizes[self.classes[node as usize] as usize] += 1;
    }

    /// Returns the class identifier of node `node` or `None` if `node` is unassigned
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.class_of_node(1), None);
    /// assert_eq!(partition.class_of_node(2), Some(class_id));
    /// ```
    pub fn class_of_node(&self, node: Node) -> Option<PartitionClass> {
        let class_id = self.classes[node as usize];
        if class_id == 0 {
            None
        } else {
            Some(class_id - 1)
        }
    }

    /// Returns the class identifier if both nodes `u` and `v` are assigned to the same class
    /// and `None` otherwise.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let c1 = partition.add_class([2,4]);
    /// let c2 = partition.add_class([6,8]);
    /// assert_eq!(partition.class_of_edge(0, 1), None); // both unassigned
    /// assert_eq!(partition.class_of_edge(0, 2), None); // 0 unassigned
    /// assert_eq!(partition.class_of_edge(4, 6), None); // assigned to different classes
    /// assert_eq!(partition.class_of_edge(2, 4), Some(c1));
    /// assert_eq!(partition.class_of_edge(4, 2), Some(c1));
    /// assert_eq!(partition.class_of_edge(6, 8), Some(c2));
    /// ```
    pub fn class_of_edge(&self, u: Node, v: Node) -> Option<PartitionClass> {
        let cu = self.class_of_node(u)?;
        let cv = self.class_of_node(v)?;
        if cu == cv {
            Some(cu)
        } else {
            None
        }
    }

    /// Returns the number of unassigned nodes
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// assert_eq!(partition.number_of_unassigned(), 10);
    /// partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_unassigned(), 8);
    /// ```
    pub fn number_of_unassigned(&self) -> Node {
        self.class_sizes[0]
    }

    /// Returns the number of nodes in class `class_id`
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,4]);
    /// assert_eq!(partition.number_in_class(class_id), 2);
    /// ```
    pub fn number_in_class(&self, class_id: PartitionClass) -> Node {
        self.class_sizes[class_id as usize + 1]
    }

    /// Returns the number of partition classes (0 if all nodes are unassigned)
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// let mut partition = Partition::new(10);
    /// assert_eq!(partition.number_of_classes(), 0);
    /// partition.add_class([2,4]);
    /// assert_eq!(partition.number_of_classes(), 1);
    /// ```
    pub fn number_of_classes(&self) -> Node {
        self.class_sizes.len() as Node - 1
    }

    /// Returns the members of a partition class in order.
    ///
    /// # Warning
    /// This operation is expensive and requires time linear in the total number of nodes, i.e. it
    /// is roughly independent of the actual size of partition class `class_id`.
    ///
    /// # Example
    /// ```
    /// use dfvs::graph::partition::*;
    /// use itertools::Itertools;;
    /// let mut partition = Partition::new(10);
    /// let class_id = partition.add_class([2,5,4]);
    /// assert_eq!(partition.members_of_class(class_id).collect_vec(), vec![2,4,5]);
    /// ```
    pub fn members_of_class(&self, class_id: Node) -> impl Iterator<Item = Node> + '_ {
        let class_id = class_id + 1;
        assert!(self.class_sizes.len() > class_id as usize);
        self.classes.iter().enumerate().filter_map(move |(i, &c)| {
            if c == class_id {
                Some(i as Node)
            } else {
                None
            }
        })
    }

    /// Splits the input graph `graph` (has to have the same number of nodes as `self`) into
    /// one subgraph per partition class; the `result[i]` corresponds to partition class `i`.
    ///
    /// ```
    /// use dfvs::graph::*;
    /// use dfvs::graph::generators::GeneratorSubstructures;
    /// use dfvs::graph::GraphEdgeEditing;
    /// let mut partition = Partition::new(7);
    /// partition.add_class([0,1,3]);
    /// partition.add_class([2,6,5]);
    ///
    /// let mut graph = AdjArray::new(7);
    /// graph.connect_path((0 as Node..7).into_iter());
    /// graph.add_edge(6, 6);
    ///
    /// let split : Vec<(AdjArray, NodeMapper)> = partition.split_into_subgraphs_as(&graph);
    /// assert_eq!(split.len(), 2);
    ///
    /// assert_eq!(split[0].1.new_id_of(0), Some(0));
    /// assert_eq!(split[0].1.new_id_of(2), None);
    /// assert_eq!(split[0].1.new_id_of(3), Some(2));
    /// assert_eq!(split[0].0.edges_vec(), vec![(0, 1)]);
    ///
    /// assert_eq!(split[1].1.new_id_of(0), None);
    /// assert_eq!(split[1].1.new_id_of(2), Some(0));
    /// assert_eq!(split[1].1.new_id_of(5), Some(1));
    /// assert_eq!(split[1].0.edges_vec(), vec![(1, 2), (2, 2)]);
    /// ```
    pub fn split_into_subgraphs_as<GI, GO, M>(&self, graph: &GI) -> Vec<(GO, M)>
    where
        GI: AdjacencyList,
        GO: GraphNew + GraphEdgeEditing,
        M: node_mapper::Setter + node_mapper::Getter,
    {
        assert_eq!(graph.len(), self.classes.len());

        // Create an empty graph and mapper with the capacity for each partition class
        let mut result = (0..self.number_of_classes())
            .into_iter()
            .map(|class_id| {
                let n = self.number_in_class(class_id);
                (GO::new(n as usize), M::with_capacity(n))
            })
            .collect_vec();

        // Iterator over all (assigned) nodes and map them into their respective subgraph
        let mut nodes_mapped_in_class = vec![0; self.number_of_classes() as usize];
        for (u, &class_id) in self.classes.iter().enumerate() {
            if class_id == 0 {
                // u is unassigned
                continue;
            }

            let class_id = (class_id - 1) as usize;

            result[class_id]
                .1
                .map_node_to(u as Node, nodes_mapped_in_class[class_id]);
            nodes_mapped_in_class[class_id] += 1;
        }

        // Iterate over all edges incident to assigned nodes
        for (u, &class_id) in self.classes.iter().enumerate().filter(|(_, &c)| c > 0) {
            let u = u as Node;
            let result_containg_u = &mut result[class_id as usize - 1];

            let mapped_u = result_containg_u.1.new_id_of(u).unwrap();

            // Iterate over all out-neighbors of u that are in the same partition class
            for v in graph
                .out_neighbors(u)
                .filter(|&v| self.classes[v as usize] == class_id)
            {
                let mapped_v = result_containg_u.1.new_id_of(v).unwrap();
                result_containg_u.0.add_edge(mapped_u, mapped_v);
            }
        }

        result
    }

    /// Shorthand for [`Partition::split_into_subgraphs_as`]
    ///
    /// ```
    /// use dfvs::graph::*;
    /// use dfvs::graph::generators::GeneratorSubstructures;
    /// use dfvs::graph::GraphEdgeEditing;
    /// let mut partition = Partition::new(7);
    /// partition.add_class([0,1,3]);
    /// partition.add_class([2,6,5]);
    ///
    /// let mut graph = AdjArray::new(7);
    /// graph.connect_path((0 as Node..7).into_iter());
    /// graph.add_edge(6, 6);
    ///
    /// let split = partition.split_into_subgraphs(&graph);
    /// assert_eq!(split.len(), 2);
    ///
    /// assert_eq!(split[0].1.new_id_of(0), Some(0));
    /// assert_eq!(split[0].1.new_id_of(2), None);
    /// assert_eq!(split[0].1.new_id_of(3), Some(2));
    /// assert_eq!(split[0].0.edges_vec(), vec![(0, 1)]);
    ///
    /// assert_eq!(split[1].1.new_id_of(0), None);
    /// assert_eq!(split[1].1.new_id_of(2), Some(0));
    /// assert_eq!(split[1].1.new_id_of(5), Some(1));
    /// assert_eq!(split[1].0.edges_vec(), vec![(1, 2), (2, 2)]);
    /// ```
    pub fn split_into_subgraphs<G>(&self, graph: &G) -> Vec<(G, NodeMapper)>
    where
        G: AdjacencyList + GraphNew + GraphEdgeEditing,
    {
        self.split_into_subgraphs_as(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::BitSet;
    use crate::random_models::gnp::generate_gnp;
    use rand::SeedableRng;

    #[test]
    fn cross_vertex_induced() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(123);

        for _ in 0..10 {
            let graph: AdjArray = generate_gnp(&mut rng, 100, 0.009);

            let mut sccs = graph.strongly_connected_components_no_singletons();
            let partition = graph.partition_into_strongly_connected_components();

            assert_eq!(sccs.len(), partition.number_of_classes() as usize);

            for (i, scc) in sccs.iter_mut().enumerate() {
                scc.sort_unstable();
                assert_eq!(
                    scc.clone(),
                    partition
                        .members_of_class(i as PartitionClass)
                        .collect_vec()
                );
            }

            let split = partition.split_into_subgraphs(&graph);

            for (i, scc) in sccs.into_iter().enumerate() {
                let bitmask = BitSet::new_all_unset_but(graph.len(), scc.iter().copied());
                let subgraph = graph.vertex_induced(&bitmask).0;

                assert_eq!(split[i].0.number_of_nodes(), subgraph.number_of_nodes());
                assert_eq!(split[i].0.number_of_edges(), subgraph.number_of_edges());
            }
        }
    }
}
