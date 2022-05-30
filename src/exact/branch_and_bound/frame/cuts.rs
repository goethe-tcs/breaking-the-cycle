use super::*;
use itertools::Itertools;
use rand::prelude::SliceRandom;

impl<G: BnBGraph> Frame<G> {
    pub(super) fn try_branch_on_simple_cuts(&mut self) -> Option<BBResult<G>> {
        if let Some((ap, _)) = compute_undirected_articulation_point(&self.graph) {
            return Some(self.branch_on_node(ap));
        }

        if let Some((u, v)) = compute_undirected_articulation_pair(&self.graph, true) {
            return Some(self.branch_on_node_group(vec![u, v]));
        }

        None
    }

    pub(super) fn try_branch_on_fancy_cuts(&mut self) -> Option<BBResult<G>> {
        if self.graph.len() < 64 {
            // We should never enter this branch for small graphs as we should have branched to the
            // matrix-solver. Nevertheless, fancy cut algorithms have weird corner cases and performance
            // issues for small graphs. Hence, we better skip them.
            return None;
        }

        // search for cuts
        if let Some(cut) = compute_undirected_cut(&self.graph, self.upper_bound.min(8) - 1) {
            return Some(self.branch_on_node_group(cut));
        }

        if let Some(cut) = self
            .graph
            .approx_min_balanced_cut(
                &mut rand::thread_rng(),
                30,
                0.25,
                Some(self.upper_bound.min(8)),
                true,
            )
            .or_else(|| {
                self.graph.approx_min_balanced_cut(
                    &mut rand::thread_rng(),
                    50,
                    15.0 / self.graph.number_of_nodes() as f64,
                    Some(self.upper_bound.min(5)),
                    true,
                )
            })
        {
            return Some(self.branch_on_node_group(cut));
        }

        None
    }
}

fn clone_undirected<G: BnBGraph>(graph: &G) -> G {
    let mut undir_graph = graph.clone();
    for u in graph.vertices() {
        for v in graph.out_only_neighbors(u) {
            undir_graph.add_edge(u, v);
        }
    }
    undir_graph
}

pub(super) fn compute_undirected_articulation_point<G: BnBGraph>(
    graph: &G,
) -> Option<(Node, bool)> {
    let undir_graph = clone_undirected(graph);

    let points = UndirectedCutVertices::new(&undir_graph).compute();
    if points.cardinality() == 0 {
        return None;
    }

    if let Some(x) = points
        .iter()
        .map(|x| x as Node)
        .filter(|&u| !graph.has_dir_edges(u))
        .max_by_key(|&u| graph.undir_degree(u))
        .or_else(|| {
            let mut candidates = points.iter().map(|x| x as Node).collect_vec();
            candidates.sort_unstable_by_key(|&u| {
                graph.in_degree(u) as usize * graph.out_degree(u) as usize
            });

            candidates.iter().rev().copied().find(|&u| {
                let mut subgraph = (graph).clone();
                subgraph.remove_edges_out_of_node(u);

                let mut sccs = StronglyConnected::new(&subgraph);
                sccs.set_include_singletons(false);
                sccs.take(2).count() > 1
            })
        })
    {
        return Some((x, true));
    }

    Some((
        points
            .iter()
            .map(|x| x as Node)
            .max_by_key(|&u| graph.in_degree(u) as usize * graph.out_degree(u) as usize)
            .unwrap(),
        false,
    ))
}

fn compute_undirected_articulation_pair<G: BnBGraph>(
    graph: &G,
    undir_nodes_only: bool,
) -> Option<(Node, Node)> {
    let undir_graph = clone_undirected(graph);

    let points = UndirectedCutVertices::new(&undir_graph).compute();
    if points.cardinality() == 0 {
        return None;
    }

    let mut undir_nodes = graph
        .vertices()
        .filter(|&u| !graph.has_dir_edges(u))
        .collect_vec();
    undir_nodes.sort_unstable_by_key(|&u| graph.number_of_nodes() - graph.undir_degree(u));

    for (i, &u) in undir_nodes.iter().enumerate() {
        let mut subgraph = undir_graph.clone();
        subgraph.remove_edges_at_node(u);

        let points = UndirectedCutVertices::new(&undir_graph).compute();
        if points.cardinality() == 0 {
            continue;
        }

        if let Some(&v) = undir_nodes
            .iter()
            .skip(i + 1)
            .find(|&&v| points[v as usize])
        {
            return Some((u, v));
        }
    }

    if undir_nodes_only {
        return None;
    }

    let mut dir_nodes = graph
        .vertices()
        .filter(|&u| graph.has_dir_edges(u))
        .collect_vec();

    dir_nodes.sort_unstable_by_key(|&u| {
        graph.len() * graph.len() - graph.in_degree(u) as usize * graph.out_degree(u) as usize
    });

    for (i, &u) in dir_nodes.iter().enumerate() {
        let mut subgraph = undir_graph.clone();
        subgraph.remove_edges_at_node(u);

        let points = UndirectedCutVertices::new(&undir_graph).compute();
        if points.cardinality() == 0 {
            continue;
        }

        if let Some(&v) = undir_nodes.iter().find(|&&v| points[v as usize]) {
            return Some((u, v));
        }

        if let Some(&v) = dir_nodes.iter().skip(i + 1).find(|&&v| points[v as usize]) {
            return Some((u, v));
        }
    }

    None
}

fn compute_undirected_cut<G: BnBGraph>(graph: &G, max_size: Node) -> Option<Vec<Node>> {
    let mut candidates: Vec<Node> = graph
        .vertices()
        .filter(|&u| graph.undir_degree(u) > 0 && !graph.has_dir_edges(u))
        .collect_vec();

    if candidates.len() < 10 {
        // we select at most a third of the candidates; for cuts of size at most 2, we have
        // our undirect ariticulation point routines.
        return None;
    }

    candidates.sort_unstable_by_key(|&u| graph.total_degree(u));

    let mut rng = rand::thread_rng();
    let n = graph.number_of_nodes();

    for _ in 0..30 {
        let mut subgraph = graph.clone();

        let super_cut = candidates
            .choose_multiple(&mut rng, (max_size as usize * 10).min(candidates.len() / 3))
            .copied()
            .collect_vec();
        subgraph.remove_edges_of_nodes(&super_cut);
        let partition = subgraph.partition_into_strongly_connected_components();
        if partition.number_of_classes() < 2
            || (0..partition.number_of_classes())
                .into_iter()
                .all(|i| partition.number_in_class(i) * 4 < n)
        {
            continue;
        }

        let largest_class = (0..partition.number_of_classes())
            .into_iter()
            .max_by_key(|&i| partition.number_in_class(i))
            .unwrap();

        let cut = super_cut
            .iter()
            .filter(|&&u| {
                graph
                    .undir_neighbors(u)
                    .any(|v| partition.class_of_node(v) == Some(largest_class))
                    && graph
                        .undir_neighbors(u)
                        .any(|v| partition.class_of_node(v) != Some(largest_class))
            })
            .copied()
            .collect_vec();

        if cut.is_empty() {
            return None;
        }

        if max_size as usize >= cut.len()
            && 2 * cut.len() * cut.len() <= partition.number_in_class(largest_class) as usize
        {
            return Some(cut);
        }
    }

    None
}
