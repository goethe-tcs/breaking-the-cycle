use super::*;
use bitintr::Pdep;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Graph8 {
    pub(super) matrix: u64,
    pub(super) n: Node,
}

const FIRST_COLUMNS: u64 = 0x0101_0101_0101_0101;
const DIAGONAL: u64 = 0x8040_2010_0804_0201;

impl<A: AdjacencyList> From<&A> for Graph8 {
    fn from(graph: &A) -> Self {
        debug_assert!(graph.len() <= 8);

        let matrix = graph
            .edges_iter()
            .map(|(u, v)| 1u64 << (8 * u + v))
            .fold(0, |a, b| a | b);

        Self {
            matrix,
            n: graph.number_of_nodes(),
        }
    }
}

impl BBGraph for Graph8 {
    type NodeMask = u8;
    type SccIterator<'a> = SCCIterator<'a, Self>;
    const CAPACITY: usize = 8;

    fn from_bbgraph<G: BBGraph>(graph: &G) -> Self
    where
        <G as BBGraph>::NodeMask: num::traits::AsPrimitive<Self::NodeMask>,
    {
        debug_assert!(graph.len() <= Graph8::CAPACITY);

        let matrix = graph
            // iterate over the first eight neighborhoods
            .vertices()
            .take(8)
            .map(|u| graph.out_neighbors(u as Node).as_())
            // shift the first entry by 0 bits, the next by 8, and so on
            .enumerate()
            .map(|(i, src)| {
                let src8: u8 = src.as_();
                (src8 as u64) << (8 * i)
            })
            // and compress them in a single 64 bit word
            .fold(0u64, |a, b| a | b);

        Graph8 {
            matrix,
            n: graph.len() as Node,
        }
    }

    fn len(&self) -> usize {
        self.n as usize
    }

    fn remove_first_node(&self) -> Self {
        Self {
            matrix: (self.matrix >> 9) & 0x007f_7f7f_7f7f_7f7f,
            n: self.n - 1,
        }
    }

    fn contract_first_node(&self) -> Self {
        let mut matrix = self.matrix;

        matrix |= ((self.matrix & FIRST_COLUMNS) * 0xff) & ((self.matrix & 0xff) * FIRST_COLUMNS);

        Self { matrix, n: self.n }.remove_first_node()
    }

    fn remove_node(&self, u: Node) -> Self {
        debug_assert!(u < self.n as Node);

        let mut matrix = self.matrix;

        // move columns
        matrix &= !(FIRST_COLUMNS << u); // zero out the u-th column
        let stay_mask = (FIRST_COLUMNS << u) - FIRST_COLUMNS; // bits that need to stay
        matrix = (matrix & stay_mask) | ((matrix & !stay_mask) >> 1);

        // move rows
        matrix &= !(0xffu64 << (8 * u)); // zero out the u-th row
        let upper_mask = !0u64 << (8 * u); // rows that need to move down
        matrix = (matrix & upper_mask) >> 8 | (matrix & !upper_mask);

        Self {
            matrix,
            n: self.n - 1,
        }
    }

    fn transitive_closure(&self) -> Self {
        let mut matrix = self.matrix;
        for _ in 0..3 {
            let mut out_matrix = matrix;
            for u in 0..8 {
                let out_neighbors = (matrix >> (8 * u)) & 0xff;
                let in_neighbors = (matrix >> u) & FIRST_COLUMNS;
                out_matrix |= out_neighbors * in_neighbors;
            }
            matrix = out_matrix;
        }

        Self { matrix, n: self.n }
    }

    fn nodes_with_loops(&self) -> u8 {
        self.matrix.pext(DIAGONAL) as u8
    }

    fn has_node_with_loop(&self) -> bool {
        self.matrix & DIAGONAL != 0
    }

    fn sccs(&self) -> Self::SccIterator<'_> {
        Self::SccIterator::new(self)
    }

    fn subgraph(&self, included_nodes: u8) -> Self {
        if included_nodes & self.nodes_mask() == self.nodes_mask() {
            return self.clone();
        }

        let n = included_nodes.count_ones();

        let extract_mask = (included_nodes as u64) * (included_nodes as u64).pdep(FIRST_COLUMNS);
        debug_assert_eq!(extract_mask.count_ones(), n * n);
        let extracted_compressed = self.matrix.pext(extract_mask);

        let scatter_mask = (FIRST_COLUMNS << n) - FIRST_COLUMNS;
        debug_assert_eq!(scatter_mask.count_ones(), 8 * n);

        Self {
            matrix: extracted_compressed.pdep(scatter_mask),
            n,
        }
    }

    fn first_node_has_loop(&self) -> bool {
        self.matrix & 1 == 1
    }

    fn nodes_mask(&self) -> u8 {
        ((1u32 << self.n) - 1) as u8
    }

    fn out_neighbors(&self, u: Node) -> u8 {
        (self.matrix >> (8 * u as usize)) as u8
    }

    fn has_all_edges(&self) -> bool {
        let mask = ((self.nodes_mask() as u64) * FIRST_COLUMNS) >> (64 - 8 * self.n);
        self.matrix == mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators::GeneratorSubstructures;
    use itertools::Itertools;

    #[test]
    fn bbgraph_from_generic_int() {
        let edges = [(0, 1), (1, 1), (2, 3), (6, 5)];
        let generic_graph = GenericIntGraph::<u32, 32>::from(&AdjListMatrix::from(&edges));
        let graph = Graph8::from_bbgraph(&generic_graph);
        assert_eq!(graph.edges(), Vec::from(edges));
    }

    super::super::bb_graph::bbgraph_tests!(Graph8, g8);
}
