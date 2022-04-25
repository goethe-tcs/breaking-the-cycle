use crate::graph::*;
pub mod flow_based;
mod single_staged;

pub use flow_based::*;
pub use single_staged::*;

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
        apply_rule_di_cliques(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_complete_node(&mut self) -> bool {
        apply_rule_complete_node(&mut self.graph, &mut self.in_fvs)
    }

    pub fn apply_pie_reduction(&mut self) -> bool {
        apply_rule_pie(&mut self.graph)
    }

    pub fn apply_dome_reduction(&mut self) -> bool {
        apply_rule_dome(&mut self.graph)
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
        let rule_di_cliques = apply_rule_di_cliques(graph, fvs);
        let pie = apply_rule_pie(graph);
        let dome = apply_rule_dome(graph);
        if !(rule_3 || rule_4 || rule_di_cliques || pie || dome) {
            break;
        }
    }

    if with_expensive_rules {
        apply_rule_5(graph, fvs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

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
        for filename in glob::glob("data/pace/exact_public/e_007.metis")
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
}
