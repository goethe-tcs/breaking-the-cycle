use super::*;
use std::io::{BufRead, ErrorKind, Write};

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
                writeln!(writer, "{} {}", u, v)?;
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

#[cfg(test)]
mod tests {
    use super::super::AdjListMatrix;
    use super::*;

    #[test]
    fn read_graph() {
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
}
