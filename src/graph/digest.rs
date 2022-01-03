use super::*;
use ::digest::{Digest, Output};

pub trait GraphDigest {
    /// Computes a Hash-Digest of a graph that is independent of the
    /// graph data structure used and returns it as a hex string.
    fn digest<D: Digest>(&self) -> String
    where
        digest::Output<D>: core::fmt::LowerHex;

    /// Computes a SHA256 digest using [GraphDigest::digest]. The
    /// returned string is 64 characters long.
    fn digest_sha256(&self) -> String {
        self.digest::<sha2::Sha256>()
    }
}

impl<G: AdjacencyList> GraphDigest for G {
    fn digest<D: Digest>(&self) -> String
    where
        digest::Output<D>: core::fmt::LowerHex,
    {
        let mut hasher = D::new();
        let mut buffer = [0u8; 8];

        let encode = |buf: &mut [u8], u: Node| {
            for (i, c) in buf.iter_mut().enumerate().take(4) {
                *c = (u >> (8 * i)) as u8;
            }
        };

        // first encode the number of nodes in the graph
        encode(&mut buffer[0..4], self.number_of_nodes());
        hasher.update(&buffer);

        // then append a sorted edge list
        let mut edges = self.edges();
        edges.sort_unstable();
        for (u, v) in edges {
            encode(&mut buffer[0..], u);
            encode(&mut buffer[4..], v);
            hasher.update(&buffer);
        }

        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
pub mod test {
    use crate::graph::{AdjArray, GraphDigest, GraphEdgeEditing, GraphNew};

    #[test]
    fn digest_sha256() {
        let mut graph = AdjArray::new(10);
        graph.add_edge(4, 3);
        graph.add_edge(1, 2);
        // computed with https://www.gnu.org/software/coreutils/sha256sum
        assert_eq!(
            graph.digest_sha256(),
            "73f9b526b0528f6a33e96b064f90dd9ad5b8fd646717d33e7ab1286361aa847a"
        );
    }
}
