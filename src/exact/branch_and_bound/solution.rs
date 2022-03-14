use super::*;
use bitintr::Pdep;

pub trait BBSolution: Sized + Clone {
    fn new() -> Self;
    fn insert_new_node(&mut self, i: Node) -> &mut Self;
    fn insert_new_nodes(&mut self, nodes: u64) -> &mut Self;
    fn cardinality(&self) -> Node;
    fn included(&self) -> Vec<Node>;
    fn included_mask(&self) -> u64;
    fn merge(&mut self, s: Self) -> &mut Self;
    fn shift_from_subgraph(&mut self, subgraph: u64) -> &mut Self;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    nodes_in_dfvs: u64,
    cardinality: u32,
}

impl BBSolution for Solution {
    fn new() -> Self {
        Self {
            nodes_in_dfvs: 0,
            cardinality: 0,
        }
    }

    /// Add a node to the solution. Precondition: The node is not in the solution yet.
    fn insert_new_node(&mut self, i: Node) -> &mut Self {
        debug_assert!(i < 64);
        debug_assert!(self.nodes_in_dfvs & (1_u64 << i as usize) == 0);
        self.nodes_in_dfvs |= 1_u64 << i as usize;
        self.cardinality += 1;
        self
    }

    /// Add new nodes to the solution. Precondition: None of the nodes are in the solution yet.
    fn insert_new_nodes(&mut self, nodes: u64) -> &mut Self {
        debug_assert!(self.nodes_in_dfvs & nodes as u64 == 0);
        self.nodes_in_dfvs |= nodes as u64;
        self.cardinality = self.nodes_in_dfvs.count_ones();

        self
    }

    fn cardinality(&self) -> Node {
        self.cardinality
    }

    fn included(&self) -> Vec<Node> {
        self.nodes_in_dfvs.iter_ones().collect()
    }

    fn merge(&mut self, s: Self) -> &mut Self {
        let included: u64 = self.nodes_in_dfvs | s.nodes_in_dfvs;

        self.nodes_in_dfvs = included;
        self.cardinality = included.count_ones();

        self
    }

    /// Assume the current solution `self` is a solution to a Graph `H` that is a subgraph of
    /// a graph `G`. Then given a bitvector `subgraph` describing the nodes of `H` in `G`,
    /// the function modifies the solution inplace to contain the nodes in `G`
    /// that correspond to the nodes in `H` that are currently contained.
    ///
    /// For example, if `H` is the subgraph of `G` containing nodes `{0, 3, 5, 7}`,
    /// and the solution currently contains nodes `{1, 2}`, then this modifies the solution
    /// to contain nodes `{3, 5}`, as the nodes `{1, 2}` in the subgraph correspond to nodes
    /// `{3, 5}` in the original graph.
    fn shift_from_subgraph(&mut self, subgraph: u64) -> &mut Self {
        self.nodes_in_dfvs = self.nodes_in_dfvs.pdep(subgraph);
        self
    }

    fn included_mask(&self) -> u64 {
        self.nodes_in_dfvs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_from_subgraph() {
        assert_eq!(
            *Solution::new().shift_from_subgraph(0b00000000),
            Solution::new()
        );

        assert_eq!(
            *Solution::new().shift_from_subgraph(0b00101011),
            Solution::new()
        );

        assert_eq!(
            *Solution::new().shift_from_subgraph(u32::MAX as u64),
            Solution::new()
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .shift_from_subgraph(0b00000001),
            *Solution::new().insert_new_node(0)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(0)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(1)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(1)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(2)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(3)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(3)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(5)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(1)
                .insert_new_node(4)
                .insert_new_node(0)
                .shift_from_subgraph(0b10101011),
            *Solution::new()
                .insert_new_node(0)
                .insert_new_node(1)
                .insert_new_node(7)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(0)
                .insert_new_node(1)
                .shift_from_subgraph(0b00101011),
            *Solution::new().insert_new_node(0).insert_new_node(1)
        );

        assert_eq!(
            *Solution::new()
                .insert_new_node(3)
                .insert_new_node(7)
                .insert_new_node(1)
                .shift_from_subgraph(0b11111111),
            *Solution::new()
                .insert_new_node(3)
                .insert_new_node(7)
                .insert_new_node(1)
        );
    }
}
