use super::*;
use std::fs::{File, OpenOptions};
use std::io::{stdout, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub use dot::DotWrite;
pub use metis::{MetisRead, MetisWrite};
pub use pace::{PaceRead, PaceWrite};

pub mod dot {
    use super::*;
    use fxhash::FxHashSet;
    use itertools::Itertools;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    pub trait DotWrite {
        /// Write out graph as DOT file where nodes included in the iterator are highlighted
        fn try_write_dot_with_solution<W: Write, D: IntoIterator<Item = Node>>(
            &self,
            writer: W,
            dfvs: D,
        ) -> std::io::Result<()>;

        /// produces a minimalistic DOT representation of the graph
        fn try_write_dot<W: Write>(&self, writer: W) -> Result<(), std::io::Error>;
    }

    const ATTRIB_NODE_IN_DFVS: &str = "[color=red]";
    const ATTRIB_EDGE_UNDIR: &str = "[dir=n,color=\"#0000ff\",penwidth=2]";
    const ATTRIB_EDGE_UNDIR_DEL: &str = "[dir=n,color=\"#a0a0ff\",penwidth=2]";
    const ATTRIB_EDGE_DIR: &str = "";
    const ATTRIB_EDGE_DIR_DEL: &str = "[color=\"#a0a0a0\"]";

    fn format_node(u: Node) -> String {
        u.to_string()
    }

    impl<G: AdjacencyList> DotWrite for G {
        fn try_write_dot_with_solution<W: Write, D: IntoIterator<Item = Node>>(
            &self,
            mut writer: W,
            dfvs: D,
        ) -> std::io::Result<()> {
            let dfvs: HashSet<Node> = dfvs.into_iter().collect();

            let undirected_edges = {
                let forward: FxHashSet<Edge> =
                    FxHashSet::from_iter(self.edges_iter().filter(|(u, v)| u < v));

                FxHashSet::from_iter(self.edges_iter().filter_map(|(u, v)| {
                    if u > v && forward.contains(&(v, u)) {
                        Some((v, u))
                    } else {
                        None
                    }
                }))
            };

            let mut statements = Vec::with_capacity(2 + 4 * self.len());
            statements.push(format!(
                "stats [label=\"n = {}\\nm = {}\\nk = {}\",labeljust=l,rank=min,penwidth=0]",
                self.number_of_nodes(),
                self.number_of_edges(),
                dfvs.len()
            ));

            // export nodes in DFVS
            {
                let mut vec = dfvs.iter().copied().collect_vec();
                vec.sort_unstable();
                statements.push(
                    vec.iter().copied().map(format_node).join(",") + " " + ATTRIB_NODE_IN_DFVS,
                );
            }

            for u in self.vertices().filter(|&u| self.out_degree(u) > 0) {
                let is_deleted = dfvs.contains(&u);

                let mut neighbors = [
                    vec![], // 0: undir
                    vec![], // 1: undir+del
                    vec![], // 2: dir
                    vec![], // 3: dir + delt
                ];

                // identify the edge type of all outgoing edges
                for v in self.out_neighbors(u) {
                    let is_undir = undirected_edges.contains(&(u.min(v), u.max(v)));
                    let is_deleted = is_deleted || dfvs.contains(&v);

                    if is_undir && u > v {
                        continue;
                    }

                    neighbors[2 * (is_undir as usize) + (is_deleted as usize)].push(v);
                }

                let mut print_neighbors = |vec: &mut Vec<Node>, attrib: &str| {
                    if vec.is_empty() {
                        return;
                    };
                    vec.sort_unstable();

                    statements.push(format!(
                        "{} -> {{{}}} {}",
                        format_node(u),
                        vec.iter().copied().map(format_node).join(" "),
                        attrib
                    ));
                };

                print_neighbors(&mut neighbors[0], ATTRIB_EDGE_UNDIR);
                print_neighbors(&mut neighbors[1], ATTRIB_EDGE_UNDIR_DEL);
                print_neighbors(&mut neighbors[2], ATTRIB_EDGE_DIR);
                print_neighbors(&mut neighbors[3], ATTRIB_EDGE_DIR_DEL);
            }

            writeln!(writer, "digraph {{\n {};\n}}", statements.join(";\n "))
        }

        fn try_write_dot<T: Write>(&self, mut writer: T) -> Result<(), std::io::Error> {
            let n = self.number_of_nodes();
            let m = self.number_of_edges();
            write!(writer, "digraph {{ /* n={} m={} */", n, m)?;
            for u in self.vertices() {
                if self.out_degree(u) == 0 {
                    continue;
                }

                write!(writer, " v{} -> {{", u)?;
                for v in self.out_neighbors(u) {
                    write!(writer, " v{}", v)?;
                }
                write!(writer, " }}; ")?;
            }
            writeln!(writer, "}}")?;
            Ok(())
        }
    }
}

pub mod metis {
    use super::*;
    use std::io::Error;

    pub trait MetisRead: Sized {
        fn try_read_metis<T: BufRead>(buf: T) -> Result<Self, std::io::Error>;
    }

    pub trait MetisWrite {
        fn try_write_metis<T: Write>(&self, writer: T) -> Result<(), std::io::Error>;
    }

    impl<G: GraphFromSlice + Sized> MetisRead for G {
        fn try_read_metis<T: BufRead>(reader: T) -> Result<Self, Error> {
            let mut non_comment_lines = reader.lines().filter_map(|x| -> Option<String> {
                if let Ok(line) = x {
                    if !line.starts_with('%') {
                        return Some(line);
                    }
                }
                None
            });

            let error = |msg| Err(std::io::Error::new(ErrorKind::Other, msg));

            // parse header
            let (n, m) = {
                if let Some(header) = non_comment_lines.next() {
                    let fields: Vec<_> = header.split(' ').collect();
                    if fields.len() != 3 {
                        return error("Expected exactly 3 header fields");
                    }

                    if fields[2].parse() != Ok(0) {
                        return error("Only support unweighted graphs");
                    }

                    let n: Node = match fields[0].parse() {
                        Ok(n) => n,
                        Err(_) => return error("Cannot parse number of nodes"),
                    };

                    let m: usize = match fields[0].parse() {
                        Ok(m) => m,
                        Err(_) => return error("Cannot parse number of edges"),
                    };

                    (n, m)
                } else {
                    return error("Cannot read header");
                }
            };

            let mut edges: Vec<(Node, Node)> = Vec::with_capacity(m);

            // read neighbors
            for (source, neighbors) in non_comment_lines.enumerate() {
                let neighbors = neighbors.trim();
                if neighbors.is_empty() {
                    continue;
                }

                if source >= n as usize {
                    return error("Too many neighborhoods");
                }

                for v in neighbors.split(' ') {
                    if let Ok(int_v) = v.parse::<Node>() {
                        if 0 == int_v || int_v > n {
                            return error(
                                format!("Neighbor {} of {} is out of range", int_v, source)
                                    .as_str(),
                            );
                        }

                        edges.push((source as Node, int_v - 1));
                    } else {
                        return error(format!("Cannot parse neighbor of {}", source).as_str());
                    }
                }
            }

            edges.sort_unstable();
            edges.dedup();

            Ok(G::from_slice(n, edges.as_slice(), true))
        }
    }

    impl<G: AdjacencyList> MetisWrite for G {
        fn try_write_metis<T: Write>(&self, mut writer: T) -> Result<(), Error> {
            writeln!(
                writer,
                "{} {} 0",
                self.number_of_nodes(),
                self.number_of_edges()
            )?;
            for u in self.vertices() {
                let neigh_str: Vec<_> =
                    self.out_neighbors(u).map(|x| (x + 1).to_string()).collect();
                writeln!(writer, "{}", neigh_str.join(" "))?;
            }
            Ok(())
        }
    }
}

pub mod pace {
    use super::*;
    pub trait PaceRead: Sized {
        fn try_read_pace<T: BufRead>(buf: T) -> Result<Self, std::io::Error>;
    }

    pub trait PaceWrite {
        fn try_write_pace<T: Write>(&self, writer: T) -> Result<(), std::io::Error>;
    }

    impl<G: AdjacencyList> PaceWrite for G {
        fn try_write_pace<T: Write>(&self, mut writer: T) -> Result<(), std::io::Error> {
            let n = self.number_of_nodes();
            let m = self.number_of_edges() as u32;
            writeln!(writer, "p dfvs {} {}", n, m,)?;
            for u in self.vertices() {
                for v in self.out_neighbors(u) {
                    writeln!(writer, "{} {}", u + 1, v + 1)?;
                }
            }
            Ok(())
        }
    }

    impl<G: GraphNew + GraphEdgeEditing + Sized> PaceRead for G {
        fn try_read_pace<T: BufRead>(reader: T) -> Result<Self, std::io::Error> {
            let mut graph: Option<Self> = None;
            let mut order: Option<usize> = None;
            for line in reader.lines() {
                let line = line?;
                let elements: Vec<_> = line.split(' ').collect();
                match elements[0] {
                    "c" => {
                        // who cares about comments..
                    }
                    "p" => {
                        order = Some(parse_order(&elements)?);
                        graph = Some(G::new(order.unwrap()))
                    }
                    _ => match graph.as_mut() {
                        Some(graph) => {
                            let u = parse_vertex(elements[0], order.unwrap())?;
                            let v = parse_vertex(elements[1], order.unwrap())?;
                            graph.add_edge(u, v);
                        }
                        None => {
                            return Err(std::io::Error::new(
                                ErrorKind::Other,
                                "Edges encountered before graph creation",
                            ));
                        }
                    },
                };
            }
            match graph {
                Some(graph) => Ok(graph),
                None => Err(std::io::Error::new(
                    ErrorKind::Other,
                    "No graph created during parsing",
                )),
            }
        }
    }

    fn parse_vertex(v: &str, order: usize) -> Result<Node, std::io::Error> {
        match v.parse::<Node>() {
            Ok(u) => {
                if u == 0 || u > order as Node {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Invalid vertex label",
                    ))
                } else {
                    Ok(u - 1)
                }
            }
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid vertex label",
            )),
        }
    }

    fn parse_order(elements: &[&str]) -> Result<usize, std::io::Error> {
        if elements.len() < 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid line received starting with p",
            ));
        }
        match elements[2].parse::<usize>() {
            Ok(order) => Ok(order),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid order of graph",
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FileFormat {
    Dot,
    Metis,
    Pace,
}
impl FromStr for FileFormat {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dot" => Ok(FileFormat::Dot),
            "metis" => Ok(FileFormat::Metis),
            "pace" => Ok(FileFormat::Pace),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown type: {}", s).as_str(),
            )),
        }
    }
}

pub trait GraphRead: Sized + PaceRead + MetisRead {
    /// Tries to read the graph file at the passed in path
    fn try_read_graph(format: FileFormat, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);

        match format {
            FileFormat::Metis => Self::try_read_metis(buf_reader),
            FileFormat::Pace => Self::try_read_pace(buf_reader),
            FileFormat::Dot => Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                "Can't read dot files",
            )),
        }
    }
}

impl<G: PaceRead + MetisRead> GraphRead for G {}

/// Intended for binaries to output a resulting graph
pub struct DefaultWriter {
    file: Option<File>,
    format: FileFormat,
}

impl DefaultWriter {
    /// If output is None, we will try to write to stdout, otherwise we will try to open the file
    pub fn from_path(output: Option<PathBuf>, format: Option<FileFormat>) -> std::io::Result<Self> {
        let detected_format = output.as_ref().and_then(|p| {
            p.extension().and_then(|ext| {
                ext.to_str()
                    .and_then(|ext_str| FileFormat::from_str(ext_str).ok())
            })
        });

        let used_format = match format {
            Some(f) => f,
            None => match detected_format {
                Some(f) => f,
                None => {
                    if output.is_none() {
                        FileFormat::Metis
                    } else {
                        return Err(std::io::Error::new(
                        ErrorKind::Other,
                        "Either select the file format explicitly or use a path with extension (.dot|.pace|.metis)",
                    ));
                    }
                }
            },
        };

        let file = match output {
            Some(path) => Some(OpenOptions::new().write(true).create(true).open(path)?),
            None => None,
        };

        Ok(Self {
            file,
            format: used_format,
        })
    }

    pub fn write<G: AdjacencyList>(self, graph: &G) -> std::io::Result<()> {
        match self.file {
            None => {
                let stdout = stdout();
                let to = stdout.lock();
                match self.format {
                    FileFormat::Dot => graph.try_write_dot(to),
                    FileFormat::Metis => graph.try_write_metis(to),
                    FileFormat::Pace => graph.try_write_pace(to),
                }?;
            }
            Some(to) => {
                match self.format {
                    FileFormat::Dot => graph.try_write_dot(to),
                    FileFormat::Metis => graph.try_write_metis(to),
                    FileFormat::Pace => graph.try_write_pace(to),
                }?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::AdjListMatrix;
    use super::*;
    use crate::random_models::gnp::generate_gnp;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn test_file_format() {
        assert_eq!(FileFormat::from_str("dot").unwrap(), FileFormat::Dot);
        assert_eq!(FileFormat::from_str("dOt").unwrap(), FileFormat::Dot);
        assert_eq!(FileFormat::from_str("metis").unwrap(), FileFormat::Metis);
        assert_eq!(FileFormat::from_str("meTis").unwrap(), FileFormat::Metis);
        assert_eq!(FileFormat::from_str("pace").unwrap(), FileFormat::Pace);
        assert_eq!(FileFormat::from_str("pAce").unwrap(), FileFormat::Pace);
        assert!(FileFormat::from_str("tot").is_err());
    }

    #[test]
    fn test_default_writer() {
        assert!(DefaultWriter::from_path(None, None).is_ok());
        assert!(DefaultWriter::from_path(Some(PathBuf::from("bla.bla")), None).is_err());
    }

    #[cfg(feature = "tempfile")]
    #[test]
    fn test_default_writer_tempfile() {
        let temp_dir = tempfile::TempDir::new().unwrap();

        let org_edges = vec![(0, 1), (1, 2), (4, 5)];
        let graph = AdjListMatrix::from(&org_edges);

        for (name, format, read_fmt) in [
            ("b-dot", Some(FileFormat::Dot), None),
            ("b-metis", Some(FileFormat::Metis), Some(FileFormat::Metis)),
            ("b-pace", Some(FileFormat::Pace), Some(FileFormat::Pace)),
            ("b.dot", None, None),
            ("b.metis", None, Some(FileFormat::Metis)),
            ("b.pace", None, Some(FileFormat::Pace)),
        ] {
            let total_path = PathBuf::from(temp_dir.path().join(name));
            assert!(total_path.ends_with(name));
            let writer = DefaultWriter::from_path(Some(total_path.clone()), format);

            assert!(
                writer.is_ok(),
                "file: {} format: {:?} error: {}",
                total_path.to_str().unwrap(),
                format,
                writer.err().unwrap()
            );

            writer.unwrap().write(&graph).unwrap();

            if let Some(fmt) = read_fmt {
                let file = File::open(total_path).unwrap();
                let reader = std::io::BufReader::new(file);

                let read_graph = match fmt {
                    FileFormat::Metis => AdjListMatrix::try_read_metis(reader).unwrap(),
                    FileFormat::Pace => AdjListMatrix::try_read_pace(reader).unwrap(),
                    _ => panic!(""),
                };

                assert_eq!(graph.number_of_nodes(), read_graph.number_of_nodes());
                assert_eq!(org_edges, read_graph.edges_vec());
            }
        }
    }

    #[test]
    fn read_metis_graph() {
        let data = "%test\n7 8 0\n%test\n2 4 5 6\n3 7\n6\n\n7 6\n".as_bytes();
        let graph: Result<AdjListMatrix, std::io::Error> = AdjListMatrix::try_read_metis(data);

        assert!(graph.is_ok(), "{}", graph.err().unwrap());
        let graph = graph.unwrap();

        assert_eq!(graph.number_of_nodes(), 7);

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(0, 3));
        assert!(graph.has_edge(0, 4));
        assert!(graph.has_edge(0, 5));
        assert!(graph.has_edge(1, 2));
        assert!(graph.has_edge(1, 6));
        assert!(graph.has_edge(2, 5));
        assert!(graph.has_edge(4, 5));
        assert!(graph.has_edge(4, 6));
    }

    #[test]
    fn read_metis_broken_inputs() {
        for buffer in [
            "",                                     // no header
            "\n",                                   // empty header
            "0 1\n",                                // too short header
            "0 1 0 2\n",                            // too long header
            "%no header",                           // no header
            "a 8 0\n2 4 5 6\n3 7\n6\n\n7 6",        // invalid nodes
            "7 8 1\n2 4 5 6\n3 7\n6\n\n7 6",        // weighted
            "6 8 0\n2 4 5 6\n3 7\n6\n\n7 6",        // to many nodes
            "7 8 0\n2 4 5 6\n3 7\n6\n\n7 6\n\n\n1", // too many lines
            "7 8 1\n2 a 5 6\n3 7\n6\n\n7 6",        // invalid neighbor
        ] {
            assert!(
                AdjListMatrix::try_read_metis(buffer.as_bytes()).is_err(),
                "Error not found in {}",
                buffer
            );
        }
    }

    #[test]
    fn read_pace_graph() {
        let data = "p dfvs 7 9\n1 2\n1 4\n1 5\n1 6\n2 3\n2 7\n3 7\n4 5\n4 6".as_bytes();
        let graph: Result<AdjListMatrix, std::io::Error> = AdjListMatrix::try_read_pace(data);

        assert!(graph.is_ok());
        let graph = graph.unwrap();

        assert_eq!(graph.number_of_nodes(), 7);

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(0, 3));
        assert!(graph.has_edge(0, 4));
        assert!(graph.has_edge(0, 5));
        assert!(graph.has_edge(1, 2));
        assert!(graph.has_edge(1, 6));
        assert!(graph.has_edge(2, 6));
        assert!(graph.has_edge(3, 4));
        assert!(graph.has_edge(3, 5));
    }

    macro_rules! round_trip_test {
        ($fnname:ident, $r:ident, $w:ident) => {
            #[test]
            fn $fnname() {
                // redirects the output of try_write_pace into Vec<u8> and then reads from the buffer
                // with try_read_pace. Test succeeds if the original graph and the read graph match.
                let mut gen = Pcg64Mcg::seed_from_u64(123);
                for i in 1..20 {
                    let graph: AdjListMatrix = generate_gnp(&mut gen, 3 * i, 0.1 / i as f64);
                    let mut buffer = vec![];
                    graph.$w(&mut buffer).unwrap();
                    let read_graph = AdjListMatrix::$r(buffer.as_slice()).unwrap();

                    assert_eq!(graph.number_of_nodes(), read_graph.number_of_nodes());
                    assert_eq!(graph.number_of_edges(), read_graph.number_of_edges());
                    assert_eq!(graph.edges_vec(), read_graph.edges_vec());
                }
            }
        };
    }

    round_trip_test!(test_metis_round_trip, try_read_metis, try_write_metis);
    round_trip_test!(test_pace_round_trip, try_read_pace, try_write_pace);
}
