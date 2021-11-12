use crate::graph::Graph;
use std::convert::TryFrom;
use std::io::{BufRead, ErrorKind, Write};

pub struct PaceReader<T: BufRead>(pub T);

pub struct PaceWriter<'a, T: Write> {
    graph: &'a Graph,
    writer: T,
}

impl<'a, T: Write> PaceWriter<'a, T> {
    pub fn new(graph: &'a Graph, writer: T) -> Self {
        Self { graph, writer }
    }
}

impl<'a, T: Write> PaceWriter<'a, T> {
    pub fn output(mut self) -> Result<(), std::io::Error> {
        let n = self.graph.order();
        let m: u32 = self
            .graph
            .vertices()
            .map(|u| self.graph.out_degree(u))
            .sum();
        writeln!(self.writer, "p dfvs {} {}", n, m,)?;
        for u in self.graph.vertices() {
            for v in self.graph.out_neighbors(u) {
                writeln!(self.writer, "{} {}", u, *v)?;
            }
        }
        Ok(())
    }
}

impl<T: BufRead> TryFrom<PaceReader<T>> for Graph {
    type Error = std::io::Error;

    fn try_from(reader: PaceReader<T>) -> Result<Self, Self::Error> {
        let reader = reader.0;
        let mut graph: Option<Graph> = None;
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
                    graph = Some(Graph::new(order.unwrap()))
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

fn parse_vertex(v: &str, order: usize) -> Result<u32, std::io::Error> {
    match v.parse::<u32>() {
        Ok(u) => {
            if u == 0 || u > order as u32 {
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
    use crate::graph::Graph;
    use crate::io::PaceReader;
    use std::convert::TryFrom;

    #[test]
    fn read_graph() {
        let data = "p dfvs 7 9\n1 2\n1 4\n1 5\n1 6\n2 3\n2 7\n3 7\n4 5\n4 6".as_bytes();
        let pc = PaceReader(data);
        let graph: Result<Graph, std::io::Error> = Graph::try_from(pc);

        assert!(graph.is_ok());
        let graph = graph.unwrap();

        assert_eq!(graph.order(), 7);

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
