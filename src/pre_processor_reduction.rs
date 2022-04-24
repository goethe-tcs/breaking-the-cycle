use crate::bitset::BitSet;
use crate::graph::network_flow::{EdmondsKarp, ResidualBitMatrix, ResidualNetwork};
use crate::graph::*;
use fxhash::FxHashSet;
use std::collections::HashSet;
use std::iter::FromIterator;

pub trait ReducibleGraph = GraphNew
    + GraphEdgeEditing
    + AdjacencyList
    + AdjacencyTest
    + AdjacencyListIn
    + AdjacencyListUndir
    + Sized
    + std::fmt::Debug
    + Clone;

/// Rule5 and Rule6 need max_nodes: Node.
/// That is used to not perform the reduction rule if the reduced graph has more than max_nodes nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rules {
    Rule1,
    Rule3,
    Rule4,
    Rule5(Node),
    Rule6(Node),
    Rule56(Node),
    DiClique,
    CompleteNode,
    PIE,
    DOME,
    STOP,
    RestartRules,
}

/// rules: contains the rules of the enum Rules.
/// scc: split the graph in strongly connected components.
/// upper_bound: set and upper_bound for rule6.
/// reduced: is the result which contains (the/all) reduced graph(s). Depends on scc = true or scc = false
/// fvs: contains all nodes that belong to a smallest dfvs over all reduced graphs
/// pre_processor: used to perform the reduction rules
pub struct SuperReducer<G> {
    rules: Vec<Rules>,
    scc: bool,
    upper_bound: Option<Node>,
    to_be_reduced: Vec<(G, NodeMapper, u32)>,
    reduced: Vec<(G, NodeMapper)>,
    fvs: Vec<Node>,
    pre_processor: PreprocessorReduction<G>,
    upper_bound_below_zero: bool,
    more_rules_to_do: bool,
}

type Fvs = Vec<Node>;
type SccGraphs<G> = Vec<(G, NodeMapper)>;

impl<G> SuperReducer<G>
where
    G: ReducibleGraph,
{
    /// creates a SuperReducer with default rules (Rule1, Rule3, Rule4) and scc = true
    /// use .reduce() to reduce the graph
    pub fn new(graph: G) -> Self {
        Self::with_settings(graph, vec![Rules::Rule1, Rules::Rule3, Rules::Rule4], true)
    }

    /// use .reduce() to reduce the graph
    pub fn with_settings(graph: G, rules: Vec<Rules>, scc: bool) -> Self {
        Self {
            rules,
            scc,
            upper_bound: None,
            to_be_reduced: vec![(graph, NodeMapper::identity(0), 0)],
            reduced: vec![],
            fvs: vec![],
            pre_processor: PreprocessorReduction::from(G::new(0)),
            upper_bound_below_zero: false,
            more_rules_to_do: false,
        }
    }

    /// reduces the given graph and returns Some(in_dfvs, vec![(graph, mapper), (graph, mapper), ...])
    /// returns None, if the upper_bound for rule 6 went below 0
    /// Example: if scc = false it returns Some(in_dfvs, vec![(graph, identity)])
    ///
    /// loops over all rules exhaustively and then calculates (if scc = true) all scc.
    /// the reduction stops, if either the reduction rules did not change the graph or
    /// only one ore less scc were calculated.
    /// when the enum STOP is used as rule in rules, then all reduction rules until the STOP are
    /// performed exhaustively. After that, all the performed rules AND the enum STOP get removed
    /// from the rules and the same process restarts.
    /// If rules contains enum rule6, but self.upper_bound = None, rule6 will be ignored.
    pub fn reduce(&mut self) -> Option<(&Fvs, &SccGraphs<G>)> {
        while let Some(mut to_be_reduced) = self.to_be_reduced.pop() {
            self.pre_processor = PreprocessorReduction::from(to_be_reduced.0);
            let mapper = to_be_reduced.1;

            loop {
                if self.upper_bound_below_zero {
                    return None;
                }
                let mut applied_rule = false;
                for index in (to_be_reduced.2 as usize)..self.rules.len() {
                    let rule = &self.rules[index];
                    applied_rule |= match *rule {
                        Rules::Rule1 => self.pre_processor.apply_rule_1(),
                        Rules::Rule3 => self.pre_processor.apply_rule_3(),
                        Rules::Rule4 => self.pre_processor.apply_rule_4(),
                        Rules::Rule5(max_nodes) => {
                            if self.pre_processor.graph().number_of_nodes() < max_nodes {
                                self.pre_processor.apply_rule_5()
                            } else {
                                false
                            }
                        }
                        Rules::Rule6(max_nodes) => {
                            if self.pre_processor.graph().number_of_nodes() < max_nodes {
                                self.update_fvs_and_upper_bound(&mapper);
                                if let Some(upper_bound) = self.upper_bound {
                                    if let Some(applied) =
                                        self.pre_processor.apply_rule_6(upper_bound)
                                    {
                                        applied
                                    } else {
                                        self.upper_bound_below_zero = true;
                                        break;
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        Rules::Rule56(max_nodes) => {
                            if self.pre_processor.graph().number_of_nodes() < max_nodes {
                                self.update_fvs_and_upper_bound(&mapper);
                                if let Some(upper_bound) = self.upper_bound {
                                    if let Some(applied) =
                                        self.pre_processor.apply_rules_5_and_6(upper_bound)
                                    {
                                        applied
                                    } else {
                                        self.upper_bound_below_zero = true;
                                        break;
                                    }
                                } else {
                                    self.pre_processor.apply_rule_5()
                                }
                            } else {
                                false
                            }
                        }
                        Rules::DiClique => self.pre_processor.apply_di_cliques_reduction(),
                        Rules::CompleteNode => self.pre_processor.apply_complete_node(),
                        Rules::PIE => self.pre_processor.apply_pie_reduction(),
                        Rules::DOME => self.pre_processor.apply_dome_reduction(),
                        Rules::STOP => {
                            if !applied_rule {
                                if index + 1 == self.rules.len()
                                    || self.rules[index + 1] == Rules::STOP
                                {
                                    break;
                                }
                                to_be_reduced.2 = (index as u32) + 1;
                                self.more_rules_to_do = true;
                            }
                            break;
                        }
                        Rules::RestartRules => {
                            if applied_rule {
                                break;
                            }
                            false
                        }
                    }
                }
                if !applied_rule {
                    break;
                }
            }
            self.update_fvs_and_upper_bound(&mapper);

            if self.scc {
                self.scc(&mapper, to_be_reduced.2);
            } else {
                self.reduced
                    .push((self.pre_processor.graph.clone(), mapper))
            }
        }
        Some((&self.fvs, &self.reduced))
    }

    pub fn get_fvs(self) -> Option<Vec<Node>> {
        if !self.upper_bound_below_zero {
            Some(self.fvs)
        } else {
            None
        }
    }

    pub fn get_reduction(self) -> Option<Vec<(G, NodeMapper)>> {
        if !self.upper_bound_below_zero {
            Some(self.reduced)
        } else {
            None
        }
    }

    fn scc(&mut self, original_mapper: &NodeMapper, index: u32) {
        let mut all_sccs_mapped = self
            .pre_processor
            .graph()
            .partition_into_strongly_connected_components()
            .split_into_subgraphs(self.pre_processor.graph());

        for (_, mapper) in &mut all_sccs_mapped {
            *mapper = NodeMapper::compose(original_mapper, mapper);
        }

        if all_sccs_mapped.len() > 1 || self.more_rules_to_do {
            self.more_rules_to_do = false;
            for (graph, mapper) in all_sccs_mapped {
                self.to_be_reduced.push((graph, mapper, index));
            }
        } else if let Some(scc) = all_sccs_mapped.pop() {
            self.reduced.push(scc);
        }
    }

    fn update_fvs_and_upper_bound(&mut self, mapper: &NodeMapper) {
        let subtract_upper_bound = self.pre_processor.in_fvs.len() as Node;
        self.upper_bound = self.upper_bound.map(|ub| ub - subtract_upper_bound);
        for node in &self.pre_processor.in_fvs {
            self.fvs.push(mapper.old_id_of(*node).unwrap());
        }
        self.pre_processor.clear_fvs();
    }

    pub fn set_upper_bound(&mut self, upper_bound: Node) {
        self.upper_bound = Some(upper_bound);
    }
}

pub trait ReductionState<G> {
    fn graph(&self) -> &G;
    fn graph_mut(&mut self) -> &mut G;

    fn fvs(&self) -> &[Node];
    fn add_to_fvs(&mut self, u: Node);
    fn clear_fvs(&mut self);
}

pub struct PreprocessorReduction<G> {
    graph: G,
    in_fvs: Vec<Node>,
}

impl<G> ReductionState<G> for PreprocessorReduction<G> {
    fn graph(&self) -> &G {
        &self.graph
    }
    fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    fn fvs(&self) -> &[Node] {
        &self.in_fvs
    }
    fn add_to_fvs(&mut self, u: Node) {
        self.in_fvs.push(u);
    }
    fn clear_fvs(&mut self) {
        self.in_fvs.clear();
    }
}

impl<G> From<G> for PreprocessorReduction<G> {
    fn from(graph: G) -> Self {
        Self {
            graph,
            in_fvs: Vec::new(),
        }
    }
}

impl<G: ReducibleGraph> PreprocessorReduction<G> {
    // added AdjacencyListIn so we can use self.graph.out_degree()
    /// applies rule 1-4 exhaustively
    /// rule 5 is optional, because it runs slow
    pub fn apply_rules_exhaustively(&mut self, with_rule_5: bool) {
        apply_rules_exhaustively(&mut self.graph, &mut self.in_fvs, with_rule_5)
    }

    pub fn apply_rule_1(&mut self) -> bool {
        apply_rule_1(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_rule_3(&mut self) -> bool {
        apply_rule_3(&mut self.graph)
    }

    pub fn apply_rule_4(&mut self) -> bool {
        apply_rule_4(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_rule_5(&mut self) -> bool {
        apply_rule_5(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_rule_6(&mut self, upper_bound: Node) -> Option<bool> {
        apply_rule_6(&mut self.graph, upper_bound, &mut self.in_fvs)
    }

    pub fn apply_rules_5_and_6(&mut self, upper_bound: Node) -> Option<bool> {
        apply_rules_5_and_6(&mut self.graph, upper_bound, &mut self.in_fvs)
    }

    pub fn apply_di_cliques_reduction(&mut self) -> bool {
        apply_di_cliques_reduction(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_complete_node(&mut self) -> bool {
        apply_complete_node(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_pie_reduction(&mut self) -> bool {
        apply_pie_reduction(&mut self.graph)
    }

    pub fn apply_dome_reduction(&mut self) -> bool {
        apply_dome_reduction(&mut self.graph)
    }
}

/// applies rule 1-4 + 5 exhaustively
/// each reduction rule returns true, if it performed a reduction. false otherwise.
/// reduction rule 5 is much slower, than rule 1-4.
/// can take a minute if graph has about 10.000 nodes.
pub fn apply_rules_exhaustively<G: ReducibleGraph>(
    graph: &mut G,
    fvs: &mut Vec<Node>,
    with_expensive_rules: bool,
) {
    apply_rule_1(graph, fvs);
    loop {
        let rule_3 = apply_rule_3(graph);
        let rule_4 = apply_rule_4(graph, fvs);
        let rule_di_cliques = apply_di_cliques_reduction(graph, fvs);
        let pie = apply_pie_reduction(graph);
        let dome = apply_dome_reduction(graph);
        if !(rule_3 || rule_4 || rule_di_cliques || pie || dome) {
            break;
        }
    }

    if with_expensive_rules {
        apply_rule_5(graph, fvs);
    }
}

/// rule 1 - self-loop
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_1<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.has_edge(u, u) {
            fvs.push(u);
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }
    applied
}

/// rule 3 sink/source nodes
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_3<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if (graph.in_degree(u) == 0) != (graph.out_degree(u) == 0) {
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }

    applied
}

/// rule 4 chaining nodes with deleting self loop
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_rule_4<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        if graph.in_degree(u) == 1 || graph.out_degree(u) == 1 {
            debug_assert!(!graph.has_edge(u, u));
            let loops = graph.contract_node(u);
            fvs.extend(&loops);
            for v in loops {
                graph.remove_edges_at_node(v);
            }
            applied = true;
        }
    }
    applied
}

/// Converts a graph into a Vector of BitSets. But every node v of the graph is split into two nodes.
/// One node has the ingoing edges of v and one has the outgoing edges of v.
/// Also adds an edge from the node with ingoing edges to the node with outgoing edges.
fn create_capacity_for_many_petals<G: ReducibleGraph>(graph: &mut G) -> Vec<BitSet> {
    let graph_len_double = graph.len() * 2;
    let mut capacity = vec![BitSet::new(graph_len_double); graph_len_double];
    for node in graph.vertices_range() {
        capacity[node as usize].set_bit(node as usize + graph.len());
        for out_node in graph.out_neighbors(node) {
            capacity[node as usize + graph.len()].set_bit(out_node as usize);
        }
    }
    capacity
}

/// Updates capacity and graph.
fn perform_petal_reduction_rule_5<G: ReducibleGraph>(
    graph: &mut G,
    node: Node,
    capacity: &mut [BitSet],
    fvs: &mut Vec<Node>,
) {
    // removing edges from capacity
    remove_edges_at_capacity_node(capacity, node, graph.number_of_nodes());
    capacity[(node + graph.len() as Node) as usize].unset_all();
    capacity[node as usize].unset_bit((node + (graph.len() as Node)) as usize);
    for bit_vector in capacity.iter_mut() {
        bit_vector.unset_bit(node as usize);
    }

    // must be collected before edges at node get removed.
    let in_neighbors: Vec<_> = graph.in_neighbors(node).collect();
    let out_neighbors: Vec<_> = graph.out_neighbors(node).collect();

    // removing edges from graph
    graph.remove_edges_at_node(node);

    // add all possible edges for (in_neighbors, out_neighbors)
    let mut loop_to_delete: Vec<Node> = vec![];
    for in_neighbor in in_neighbors {
        if out_neighbors.contains(&in_neighbor) {
            loop_to_delete.push(in_neighbor);
            continue;
        }
        for &out_neighbor in &out_neighbors {
            let edge_from = (in_neighbor as usize) + graph.len();
            let edge_to = out_neighbor as usize;
            capacity[edge_from].set_bit(edge_to); // adds new edge to capacity

            graph.try_add_edge(in_neighbor, out_neighbor); // adds new edge to graph
        }
    }
    for node in loop_to_delete {
        fvs.push(node);
        graph.remove_edges_at_node(node);
        remove_edges_at_capacity_node(capacity, node, graph.number_of_nodes());
    }
}

fn count_petals(
    node: Node,
    graph_size: usize,
    mut capacity: Vec<BitSet>,
    mut labels: Vec<Node>,
    count_up_to: Node,
) -> (Vec<BitSet>, Vec<Node>, usize) {
    // prepare the capacity to run num_disjoint()
    capacity[node as usize].unset_bit(node as usize + graph_size);
    let petal_bit_matrix = ResidualBitMatrix::from_capacity_and_labels(
        capacity,
        labels,
        node + graph_size as Node,
        node,
    );

    let mut ec = EdmondsKarp::new(petal_bit_matrix);
    ec.set_remember_changes(true);
    let petal_count = ec.count_num_disjoint_upto(count_up_to) as usize;
    ec.undo_changes();
    (capacity, labels) = ec.take(); // needed because capacity and labels is moved when petal_bit_matrix is created
    (capacity, labels, petal_count)
}

/// Checks for each node u on a given graph, if u has exactly one petal.
/// If so, the graph gets reduced. That means, the node u with petal=1 gets deleted and all possible
/// edges (in_neighbors(u), out_neighbour(u)) are added to the graph.
pub fn apply_rule_5<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;

    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        let petal_count;
        (capacity, labels, petal_count) = count_petals(node, graph.len(), capacity, labels, 2);

        // actual reduction of graph (if petal_count = 1)
        if petal_count == 1 {
            applied = true;
            perform_petal_reduction_rule_5(graph, node as Node, &mut capacity, fvs);
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }
    applied
}

pub fn apply_rule_6<G: ReducibleGraph>(
    graph: &mut G,
    upper_bound: Node,
    fvs: &mut Vec<Node>,
) -> Option<bool> {
    let num_nodes = graph
        .vertices()
        .filter(|&u| graph.total_degree(u) > 0)
        .count() as Node;
    if num_nodes < upper_bound {
        return Some(false);
    }
    let mut applied_counter = upper_bound;
    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        if graph.in_degree(node) < applied_counter || graph.out_degree(node) < applied_counter {
            continue;
        };
        let petal_count;
        (capacity, labels, petal_count) =
            count_petals(node, graph.len(), capacity, labels, applied_counter + 1);

        // actual reduction of graph (if petal_count > upper_bound)
        if petal_count > applied_counter as usize {
            fvs.push(node);

            // removing edges from capacity
            remove_edges_at_capacity_node(&mut capacity, node, graph.number_of_nodes());

            // removing edges from graph
            graph.remove_edges_at_node(node);
            if applied_counter == 0 {
                return None;
            }
            applied_counter -= 1;
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }

    Some(applied_counter < upper_bound)
}

pub fn apply_rules_5_and_6<G: ReducibleGraph>(
    graph: &mut G,
    upper_bound_incl: Node,
    fvs: &mut Vec<Node>,
) -> Option<bool> {
    let mut applied = false;
    let mut upper_bound_excl = upper_bound_incl + 1;

    let num_nodes = graph
        .vertices()
        .filter(|&u| graph.in_degree(u) > 0 && graph.out_degree(u) > 0)
        .take(upper_bound_excl as usize)
        .count() as Node;

    if num_nodes < upper_bound_excl
        || !graph.vertices().any(|u| {
            graph.in_degree(u) >= upper_bound_excl && graph.out_degree(u) >= upper_bound_excl
        })
    {
        return Some(apply_rule_5(graph, fvs));
    }

    let mut capacity = create_capacity_for_many_petals(graph); // this capacity is used to calculate petals for every node of graph
    let mut labels: Vec<Node> = graph // same with labels
        .vertices_range()
        .chain(graph.vertices_range())
        .collect();

    // check for each node of graph, if the reduction can be applied. If yes -> reduce
    for node in graph.vertices_range() {
        if graph.in_degree(node) == 0 || graph.out_degree(node) == 0 {
            continue;
        }

        let petal_count = if graph.undir_degree(node) >= upper_bound_excl {
            upper_bound_excl as usize
        } else {
            let mut count_to = upper_bound_excl
                .min(graph.in_degree(node))
                .min(graph.out_degree(node));

            if count_to == 0 {
                continue;
            }

            if count_to < upper_bound_excl {
                count_to = 2;
            }

            let petal_count;
            (capacity, labels, petal_count) =
                count_petals(node, graph.len(), capacity, labels, count_to);
            petal_count
        };

        if petal_count == 1 {
            let fvs_len_before = fvs.len();
            perform_petal_reduction_rule_5(graph, node as Node, &mut capacity, fvs);
            upper_bound_excl -= (fvs.len() - fvs_len_before) as Node;
            applied = true;
            continue;
        } else if petal_count >= upper_bound_excl as usize {
            fvs.push(node);

            // removing edges from capacity
            remove_edges_at_capacity_node(&mut capacity, node, graph.len() as Node);

            // removing edges from graph
            graph.remove_edges_at_node(node);
            if upper_bound_excl == 0 {
                return None;
            }
            upper_bound_excl -= 1;
            applied = true;
            continue;
        }

        capacity[node as usize].set_bit(node as usize + graph.len());
    }

    Some(applied)
}

fn remove_edges_at_capacity_node(capacity: &mut [BitSet], node: Node, graph_size: Node) {
    capacity[(node + graph_size) as usize].unset_all();
    capacity[node as usize].unset_bit((node + (graph_size)) as usize);
    for bit_vector in capacity.iter_mut() {
        bit_vector.unset_bit(node as usize);
    }
}

/// Safe Di-Cliques Reduction, requires self loops to be deleted
///
/// returns true if rule got applied at least once, false if not at all
pub fn apply_di_cliques_reduction<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        let k = graph.undir_degree(u);

        if k == 0 || graph.undir_neighbors(u).any(|v| graph.undir_degree(v) < k) {
            continue;
        }

        let neighbors: HashSet<Node, _> = FxHashSet::from_iter(graph.undir_neighbors(u));

        if neighbors.iter().any(|&v| {
            graph
                .undir_neighbors(v)
                .filter(|x| *x == u || neighbors.contains(x))
                .take(k as usize)
                .count()
                < k as usize
        }) {
            continue;
        }

        // if u is only in 1 clique we check here if we can safely delete all nodes but u from that
        // clique. Either u has only out- or in-going edges or there is no circle back to u
        // outside of the clique. This last point is checked via breadth first search
        if graph.in_degree(u) == k
            || graph.out_degree(u) == k
            || !graph.is_node_on_cycle_after_deleting(u, neighbors.iter().copied())
        {
            for &node in &neighbors {
                graph.remove_edges_at_node(node);
            }
            fvs.extend(neighbors.iter());
            graph.remove_edges_at_node(u);
            applied = true;
        }
    }
    applied
}

pub fn apply_complete_node<G: ReducibleGraph>(graph: &mut G, fvs: &mut Vec<Node>) -> bool {
    let mut outer_applied = false;
    let mut num_nodes = graph
        .vertices()
        .filter(|&u| graph.total_degree(u) > 0)
        .count() as Node;

    loop {
        let mut applied = false;
        for node in graph.vertices_range() {
            if graph.out_degree(node) == graph.in_degree(node)
                && graph.out_degree(node) == (num_nodes - (!graph.has_edge(node, node)) as Node)
                && !graph.has_edge(node, node)
            {
                fvs.push(node);
                graph.remove_edges_at_node(node);
                applied = true;
                num_nodes -= 1;
            }
        }
        outer_applied |= applied;
        if !applied {
            return outer_applied;
        }
    }
}

/// PIE reduction rule
///
/// Looks for directed edges that are not strongly connected after removing undirected edges
pub fn apply_pie_reduction<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    let mut graph_minus_pie = AdjArray::new(graph.len());
    // get directed out_neighbors for every node and create a graph with only directed edges
    for node in graph.vertices_range() {
        for out_neighbor in graph.out_only_neighbors(node) {
            graph_minus_pie.add_edge(node, out_neighbor);
        }
    }
    let graph_sccs = graph_minus_pie.partition_into_strongly_connected_components();
    if graph_sccs.number_of_classes() == 1 && graph_sccs.number_of_unassigned() == 0 {
        return applied;
    }
    // check for every edge of our graph without undirected edges if the nodes have
    // stayed in the same scc, if not we can delete the directed edge
    for (u, v) in graph_minus_pie.edges_iter() {
        if graph_sccs.class_of_node(u) == None
            || graph_sccs.class_of_node(v) == None
            || graph_sccs.class_of_node(u) != graph_sccs.class_of_node(v)
        {
            graph.remove_edge(u, v);
            applied = true;
        }
    }
    applied
}

/// DOME reduction rule
///
/// Looks for directed edges that get dominated by other edges (un/directed edges)
pub fn apply_dome_reduction<G: ReducibleGraph>(graph: &mut G) -> bool {
    let mut applied = false;
    for u in graph.vertices_range() {
        loop {
            if graph.in_degree(u) == 0 || graph.out_degree(u) == 0 {
                break;
            }

            let in_only_neigh_u = graph.in_only_neighbors(u).collect::<FxHashSet<_>>();
            let out_neigh_u = graph.out_neighbors(u).collect::<FxHashSet<_>>();

            let v = graph.out_only_neighbors(u).find(|&neighbor|
                    // test if in_only_neigh_u is subset of in_only_neighbors(neighbor)
                    graph
                        .in_neighbors(neighbor)
                        .filter(|v| in_only_neigh_u.contains(v))
                        .take(in_only_neigh_u.len())
                        .count()
                        == in_only_neigh_u.len()
                    // test if out_only_neighbors(neighbor) is subset of out_neigh_u
                    || graph
                        .out_only_neighbors(neighbor)
                        .all(|v| out_neigh_u.contains(&v)));

            if let Some(v) = v {
                applied = true;
                graph.remove_edge(u, v);
            } else {
                break;
            }
        }
    }
    applied
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::io::MetisRead;
    use glob::glob;
    use std::fs::File;
    use std::io::BufReader;

    fn create_test_pre_processor() -> PreprocessorReduction<AdjArrayUndir> {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (1, 4),
            (2, 2),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 3),
            (4, 0),
            (5, 4),
        ]);

        let test_pre_process = PreprocessorReduction::from(graph);
        test_pre_process
    }

    fn create_test_pre_processor_2() -> PreprocessorReduction<AdjArrayUndir> {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 5),
            (3, 2),
            (3, 1),
            (3, 4),
            (4, 1),
            (4, 3),
            (4, 5),
            (5, 2),
            (5, 3),
        ]);

        let test_pre_process = PreprocessorReduction::from(graph);
        test_pre_process
    }

    fn create_test_pre_processor3() -> PreprocessorReduction<AdjArrayUndir> {
        let graph = AdjArrayUndir::from(&[(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 3), (2, 2)]);

        let test_pre_process = PreprocessorReduction::from(graph);
        test_pre_process
    }

    #[test]
    fn rule_1() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_1();

        assert_eq!(test_pre_process.graph.number_of_edges(), 4);
        assert_eq!(test_pre_process.in_fvs.len(), 2);
    }

    #[test]
    fn rule_3() {
        let mut test_pre_process = create_test_pre_processor();
        test_pre_process.apply_rule_3();
        assert_eq!(test_pre_process.graph.number_of_edges(), 8);
        assert_eq!(test_pre_process.graph.out_degree(5), 0);
    }

    #[test]
    fn rule_4_neighbor_is_neighbor() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 3);
        assert_eq!(test_pre_process.graph.number_of_edges(), 0);
    }

    #[test]
    fn rule_4_neighbor_not_neighbor() {
        let graph = AdjArrayUndir::from(&[
            (0, 2),
            (0, 5),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 5),
            (3, 2),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 5),
            (5, 2),
            (5, 3),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        test_pre_process.apply_rule_4();
        assert_eq!(test_pre_process.in_fvs.len(), 0);
        assert_eq!(test_pre_process.graph.number_of_edges(), 10);
    }

    #[test]
    fn use_rules_exhaustively() {
        let mut test_pre_process = create_test_pre_processor_2();
        test_pre_process.apply_rules_exhaustively(false);
        assert!(!test_pre_process.apply_rule_1());
        assert!(!test_pre_process.apply_rule_3());
        assert!(!test_pre_process.apply_rule_4());
    }

    fn create_circle(graph_size: usize) -> AdjArrayUndir {
        let mut graph = AdjArrayUndir::new(graph_size);
        for i in 0..graph_size - 1 {
            graph.add_edge(i as Node, (i + 1) as Node);
        }
        graph.add_edge((graph_size - 1) as Node, 0);
        graph
    }

    /// edges go in both directions
    fn create_star(satellite_count: usize) -> AdjArrayUndir {
        let mut graph = AdjArrayUndir::new(satellite_count + 1);
        for satellite in 1..satellite_count + 1 {
            graph.add_edge(0, satellite as Node);
            graph.add_edge(satellite as Node, 0);
        }
        graph
    }

    #[test]
    fn create_capacity_for_petals() {
        let tested_graph_len = 8;
        let one_less = (tested_graph_len - 1) as usize;
        let mut graph = create_circle(tested_graph_len);

        let capacity = create_capacity_for_many_petals(&mut graph);
        assert_eq!(capacity.len(), graph.len() * 2);
        let mut expected_edges = vec![];
        for node in 0..graph.len() - 1 {
            expected_edges.push((node, node + graph.len()));
            expected_edges.push((node + graph.len(), node + 1));
        }

        expected_edges.push((one_less, one_less + graph.len()));
        expected_edges.push((one_less + graph.len(), 0));

        for from_node in 0..capacity.len() {
            for to_node in 0..capacity.len() {
                if expected_edges.contains(&(from_node, to_node)) {
                    assert!(capacity[from_node][to_node]);
                } else {
                    assert!(!capacity[from_node][to_node]);
                }
            }
        }
    }

    #[test]
    fn petal_reduction() {
        let tested_graph_len = 4;
        let mut graph = create_circle(tested_graph_len);
        let mut capacity = create_capacity_for_many_petals(&mut graph);
        let mut fvs = vec![];
        perform_petal_reduction_rule_5(&mut graph, 0, &mut capacity, &mut fvs);
        let expected_edges_capacity = vec![(1, 5), (2, 6), (3, 7), (7, 1), (5, 2), (6, 3)];

        let expected_edges_graph = vec![(1, 2), (2, 3), (3, 1)];

        assert_eq!(0, graph.in_degree(0) + graph.out_degree(0));
        for from_node in 0..capacity.len() {
            for to_node in 0..capacity.len() {
                if expected_edges_capacity.contains(&(from_node, to_node)) {
                    assert!(capacity[from_node][to_node]);
                } else {
                    assert!(!capacity[from_node][to_node]);
                }
            }
        }
        for from_node in 0..graph.len() {
            for to_node in 0..graph.len() {
                if expected_edges_graph.contains(&(from_node, to_node)) {
                    assert!(graph.has_edge(from_node as Node, to_node as Node));
                } else {
                    assert!(!graph.has_edge(from_node as Node, to_node as Node));
                }
            }
        }
    }

    #[test]
    fn rule_5() {
        let tested_graph_len = 4;
        let graph = create_circle(tested_graph_len);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        test_pre_process.apply_rule_5();

        assert_eq!(test_pre_process.fvs(), [3]);
        assert_eq!(test_pre_process.fvs().len(), 1)
    }

    #[test]
    fn rule_6() {
        let graph = create_star(5);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert!(test_pre_process.apply_rule_6(3).unwrap());
        for node in test_pre_process.graph.vertices_range() {
            assert_eq!(test_pre_process.graph.total_degree(node), 0);
        }
        assert!(test_pre_process.in_fvs.contains(&0));

        assert!(!test_pre_process.apply_rule_6(0).unwrap());
        assert!(!test_pre_process.apply_rule_6(1).unwrap());
        assert!(!test_pre_process.apply_rule_6(0).unwrap());
        assert!(test_pre_process.in_fvs.contains(&0));
        assert_eq!(test_pre_process.in_fvs.len(), 1);
    }

    #[test]
    fn di_cliques_reduction() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 3),
            (0, 6),
            (1, 3),
            (1, 4),
            (1, 6),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 6),
            (5, 2),
            (5, 4),
            (5, 6),
            (6, 0),
            (6, 3),
            (6, 4),
            (6, 7),
            (6, 8),
            (7, 6),
            (7, 8),
            (8, 6),
            (8, 7),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);
        test_pre_process.apply_di_cliques_reduction();
        let mut fvs = Vec::from(test_pre_process.fvs());
        fvs.sort();
        assert_eq!(fvs, vec![1, 4, 5, 6, 8]);
    }

    fn graph_edges_count<
        G: GraphNew + GraphEdgeEditing + AdjacencyList + AdjacencyTest + AdjacencyListIn,
    >(
        graph: &G,
    ) -> Node {
        let mut edges_counter = 0;
        for node in graph.vertices_range() {
            edges_counter += graph.out_degree(node);
        }
        edges_counter
    }

    #[test]
    fn super_reducer() -> std::io::Result<()> {
        for filename in glob("data/pace/exact_public/e_007.metis")
            .unwrap()
            .filter_map(Result::ok)
        {
            let file_in = File::open(filename.as_path())?;
            let buf_reader = BufReader::new(file_in);
            let graph = AdjArrayUndir::try_read_metis(buf_reader)?;
            let mut super_reducer = SuperReducer::with_settings(
                graph,
                vec![
                    Rules::Rule1,
                    Rules::Rule3,
                    Rules::Rule4,
                    Rules::STOP,
                    Rules::Rule1,
                    Rules::Rule3,
                    Rules::Rule4,
                    Rules::DiClique,
                ],
                true,
            );
            let result = super_reducer.reduce().unwrap();
            assert_eq!(result.0.len(), 270);
            assert_eq!(result.1.len(), 2);
            assert_eq!(result.1[0].0.len(), 4);
            assert_eq!(result.1[1].0.len(), 8);
            assert_eq!(graph_edges_count(&result.1[0].0), 10);
            assert_eq!(graph_edges_count(&result.1[1].0), 40);
        }
        Ok(())
    }

    #[test]
    fn complete_node() {
        let mut pre_processor = create_test_pre_processor3();
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(0, 0);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(0, 3);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.add_edge(3, 0);
        assert!(!pre_processor.apply_complete_node());

        pre_processor.graph.remove_edge(0, 0);
        assert!(pre_processor.apply_complete_node());
    }

    #[test]
    fn pie_reduction() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 3),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 4),
            (2, 5),
            (3, 0),
            (3, 4),
            (4, 1),
            (4, 5),
            (4, 2),
            (5, 3),
            (5, 0),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 16);
        test_pre_process.apply_pie_reduction();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 14);
    }

    #[test]
    fn pie_reduction_nones() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 0),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 8);
        test_pre_process.apply_pie_reduction();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 6);
    }

    #[test]
    fn dome_reduction() {
        let graph = AdjArrayUndir::from(&[
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 5),
            (4, 3),
            (4, 5),
            (5, 0),
            (5, 4),
        ]);

        let mut test_pre_process = PreprocessorReduction::from(graph);

        assert_eq!(test_pre_process.graph.edges_vec().len(), 13);
        test_pre_process.apply_dome_reduction();
        assert_eq!(test_pre_process.graph.edges_vec().len(), 9);
    }
}
