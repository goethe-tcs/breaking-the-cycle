use std::io::{BufRead, ErrorKind};
use std::convert::TryFrom;
use petgraph::matrix_graph::{DiMatrix, NodeIndex};

pub struct PaceReader<T: BufRead>(pub T);

impl<T: BufRead> TryFrom<PaceReader<T>> for DiMatrix<(),(), Option<()>, u32>  {
    type Error = std::io::Error;

    fn try_from(reader: PaceReader<T>) -> Result<Self, Self::Error> {
        let reader = reader.0;
        let mut graph: Option<DiMatrix<_, _, _, u32>> = None;
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
                    let mut g = DiMatrix::with_capacity(order.unwrap());
                    for _ in 0..order.unwrap() {
                        g.add_node(());
                    }
                    graph = Some(g)
                }
                _ => match graph.as_mut() {
                    Some(graph) => {
                        let u = parse_vertex(elements[0], order.unwrap() as u32)?;
                        let v = parse_vertex(elements[1], order.unwrap() as u32)?;
                        graph.add_edge(NodeIndex::from(u), NodeIndex::from(v), ());
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

fn parse_vertex(v: &str, order: u32) -> Result<u32, std::io::Error> {
    match v.parse::<u32>() {
        Ok(u) => {
            if u == 0 || u > order {
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
mod tests{
    use std::convert::TryFrom;
    use petgraph::matrix_graph::{DiMatrix, NodeIndex};
    use crate::io::PaceReader;

    #[test]
    fn read_graph() {
        let data = "p tw 7 9\n1 2\n1 4\n1 5\n1 6\n2 3\n2 7\n3 7\n4 5\n4 6".as_bytes();
        let pc = PaceReader(data);
        let graph: Result<DiMatrix<_, _, _, _>, std::io::Error> = DiMatrix::try_from(pc);

        assert!(graph.is_ok());
        let graph = graph.unwrap();


        assert_eq!(graph.node_count(), 7);
        assert_eq!(graph.edge_count(), 9);

        assert!(graph.has_edge(NodeIndex::from(0), NodeIndex::from(1)));
        assert!(graph.has_edge(NodeIndex::from(0), NodeIndex::from(3)));
        assert!(graph.has_edge(NodeIndex::from(0), NodeIndex::from(4)));
        assert!(graph.has_edge(NodeIndex::from(0), NodeIndex::from(5)));
        assert!(graph.has_edge(NodeIndex::from(1), NodeIndex::from(2)));
        assert!(graph.has_edge(NodeIndex::from(1), NodeIndex::from(6)));
        assert!(graph.has_edge(NodeIndex::from(2), NodeIndex::from(6)));
        assert!(graph.has_edge(NodeIndex::from(3), NodeIndex::from(4)));
        assert!(graph.has_edge(NodeIndex::from(3), NodeIndex::from(5)));
    }
}