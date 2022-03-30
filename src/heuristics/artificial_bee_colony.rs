use crate::bitset::BitSet;
use crate::graph::*;
use crate::heuristics::utils::set_vec::HashSetVec;
use itertools::Itertools;
use rand::Rng;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::{DerefMut, IndexMut};

/// Used to calculate a heuristic solution for a dfvs of a given graph.
/// Create an ArtificialBeeColony with ArtificialBeeColony::new(...)
/// Then use run_abc() on the new ArtificialBeeColony. That returns a dfvs.
/// The ArtificialBeeColony::new(...) method explains the expected parameters.
/// The same ArtificialBeeColony should not be used to calculate a dfvs a second time.

/// # Example
/// ```
/// use dfvs::heuristics::artificial_bee_colony::*;
/// use rand::SeedableRng;
/// use rand_pcg::Pcg64;
/// use dfvs::graph::adj_array::AdjArrayIn;
///
/// let edges = &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 2)];
/// let graph = AdjArrayIn::from(edges);
/// let rand = &mut Pcg64::seed_from_u64(123);
/// let mut abc: ArtificialBeeColony<_, UndoableSolution<_, AddOneNodeStrategy>, _> =
///     ArtificialBeeColony::new(&graph, 5, 10, 5, 5, 10, rand);
/// let dfvs = abc.run_abc();
/// ```

pub trait BeeSolution<'a, G>: Clone
where
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone + SubGraph,
{
    /// perform a modification on the solution and return true if the modification was successful. return false otherwise
    fn explore<R: Rng>(&mut self, rand: &mut R) -> bool;

    /// how often in a row explore returned false
    fn get_fails(&self) -> usize;

    fn reset_fails(&mut self);

    fn included_nodes(&self) -> Vec<Node>;

    /// fitness of solution
    fn evaluate_fitness(&self) -> usize;

    fn from_all_nodes_but_fvs(graph: &'a G, in_solution: BitSet) -> Self;

    /// creates a population with population_size empty solutions
    fn create_population_simple(graph: &'a G, population_size: usize) -> Vec<Self> {
        vec![Self::from_all_nodes_but_fvs(graph, BitSet::new(graph.len())); population_size]
    }
}

impl<'a, G, St> BeeSolution<'a, G> for UndoableSolution<'a, G, St>
where
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    St: UndoableStrategy<G>,
{
    fn explore<R: Rng>(&mut self, rand: &mut R) -> bool {
        self.try_modify(rand)
    }

    fn get_fails(&self) -> usize {
        self.fails
    }

    fn reset_fails(&mut self) {
        self.fails = 0;
    }

    fn included_nodes(&self) -> Vec<Node> {
        self.included.to_vec()
    }

    fn evaluate_fitness(&self) -> usize {
        self.included.cardinality()
    }

    /// full_graph: The graph which the solution is a part of
    /// solution_graph: The solution represented as a Graph of type G (subgraph of full_graph)
    /// solution_nodes: A vector which saves for all nodes of full_graph if a node is part of the solution (true) or not (false)
    /// not_included: A vector that contains all nodes that are not in the solution
    /// fails: The count of how often in a row a solution was not acyclic after a modification
    fn from_all_nodes_but_fvs(graph: &'a G, in_solution: BitSet) -> Self {
        let mut not_included = HashSetVec::new();
        (0..graph.len())
            .filter(|&node| !in_solution[node])
            .for_each(|node| {
                not_included.insert(node as Node);
            });

        Self {
            full_graph: graph,
            solution_graph: graph.sub_graph(&in_solution),
            included: in_solution,
            not_included,
            fails: 0,
            strategy: Some(St::new()),
        }
    }
}

pub struct ArtificialBeeColony<'a, G, S, R> {
    graph: &'a G,
    population: Vec<S>,
    max_fails: usize,
    best_solution: Vec<Node>,
    old_solutions: VecDeque<S>,
    old_solutions_max: usize,
    try_keep_min_solutions: usize,
    old_solution_max_dif_to_best_solution: usize,
    old_solution_min_size: usize,
    rand: RefCell<R>,
}

impl<'a, G, S, R> ArtificialBeeColony<'a, G, S, R>
where
    G: AdjacencyListIn + GraphEdgeEditing + GraphOrder + Clone,
    S: BeeSolution<'a, G> + Clone,
    R: Rng,
{
    /// It is useful to first read the documentation of Solution::new()
    /// population: Vector of solutions
    /// max_fails: Used to decide when a solutions gets removed from the population
    /// old_solutions: Remembers solutions that were calculated after modifications
    /// old_solutions_max: Limits the size of old_solutions
    /// try_keep_min_solutions: If population gets smaller, then add a solution form old_solutions (if possible)
    /// old_solution_max_dif_to_best_solution: The maximum difference between a solutionsize and the best solution size, that is required to add a solution to old_solutions
    /// old_solution_min_size: Similar to old_solution_max_dif_to_best_solution
    pub fn new(
        graph: &'a G,
        population_size: usize,
        max_fails: usize,
        old_solutions_max: usize,
        old_solution_max_dif_to_best_solution: usize,
        old_solution_min_size: usize,
        rand: R,
    ) -> Self {
        Self {
            graph,
            population: S::create_population_simple(graph, population_size),
            max_fails,
            best_solution: vec![],
            old_solutions: VecDeque::new(),
            old_solutions_max,
            try_keep_min_solutions: 5,
            old_solution_max_dif_to_best_solution,
            old_solution_min_size,
            rand: RefCell::new(rand),
        }
    }

    fn _try_keep_min_solutions(&mut self, try_keep_min_solutions: usize) {
        self.try_keep_min_solutions = try_keep_min_solutions;
    }

    fn update_population(&mut self) {
        let population_len = self.population.len();
        for solution_index in (0..population_len).rev() {
            let solution = self.population.index_mut(solution_index);
            if !solution.explore(self.rand.borrow_mut().deref_mut()) {
                if solution.get_fails() >= self.max_fails {
                    self.population.swap_remove(solution_index);
                    if self.population.len() < self.try_keep_min_solutions {
                        if let Some(old_solution) = self.old_solutions.pop() {
                            self.population.push(old_solution);
                        }
                    }
                }
            } else {
                solution.reset_fails();

                let solution_fitness = solution.evaluate_fitness();
                if self.best_solution.len() >= self.old_solution_max_dif_to_best_solution
                    && solution_fitness
                        > self.best_solution.len() - self.old_solution_max_dif_to_best_solution
                    && solution_fitness > self.old_solution_min_size
                {
                    self.old_solutions.push(solution.clone());
                }

                if solution_fitness > self.best_solution.len() {
                    self.best_solution = solution.included_nodes().clone();
                }
            }
            if self.old_solutions.len() > self.old_solutions_max {
                self.old_solutions.pop_front();
            }
        }
    }

    pub fn run_abc(&mut self) -> Vec<Node> {
        while !self.population.is_empty() {
            self.update_population();
        }

        to_dfvs(&self.best_solution, self.graph.len())
    }
}

/// needed because a solution contains all nodes that are NOT in the dfvs
fn to_dfvs(dfvs_complement: &[Node], n: usize) -> Vec<Node> {
    (0..n as Node)
        .filter(|node| !dfvs_complement.contains(node))
        .collect_vec()
}

#[derive(Clone)]
pub struct UndoableSolution<'a, G, St> {
    full_graph: &'a G,
    solution_graph: G,
    included: BitSet,
    not_included: HashSetVec<Node>,
    fails: usize,
    strategy: Option<St>,
}

impl<'a, G, St> UndoableSolution<'a, G, St>
where
    G: GraphEdgeEditing + GraphOrder + Clone + AdjacencyListIn,
    St: UndoableStrategy<G>,
{
    fn add_fail(&mut self) {
        self.fails += 1;
    }

    fn add_node(&mut self, added_node: Node) {
        let old_val = self.included.set_bit(added_node as usize);
        debug_assert!(!old_val);

        for neighbour in self.full_graph.out_neighbors(added_node) {
            if self.included[neighbour as usize] {
                self.solution_graph.add_edge(added_node, neighbour);
            }
        }
        for neighbour in self.full_graph.in_neighbors(added_node) {
            if self.included[neighbour as usize] {
                self.solution_graph.add_edge(neighbour, added_node);
            }
        }

        self.not_included.remove(&added_node);
    }

    fn remove_node(&mut self, removed_node: Node) {
        self.not_included.insert(removed_node);
        let old_val = self.included.unset_bit(removed_node as usize);
        debug_assert!(old_val);
        self.solution_graph.remove_edges_at_node(removed_node);
    }

    /// adding and/or removing nodes to/from the solution
    fn modify(&mut self, added_nodes: &[Node], removed_nodes: &[Node]) {
        added_nodes.iter().for_each(|&node| self.add_node(node));
        removed_nodes
            .iter()
            .for_each(|&node| self.remove_node(node));
    }

    /// Can only be performed one time after a modification. Only the last modification is remembered
    fn undo_modify(&mut self, added_nodes: &Vec<Node>, removed_nodes: &Vec<Node>) {
        for removed_node in removed_nodes {
            self.add_node(*removed_node);
        }
        for added_node in added_nodes {
            self.remove_node(*added_node);
        }
    }

    /// Expects more nodes to be added than removed.
    fn try_modify<R: Rng>(&mut self, rand: &mut R) -> bool {
        let mut strategy = self.strategy.take().expect("unexpected recursive call");
        let (added_nodes, removed_nodes) = strategy.compute(self, rand);
        self.strategy = Some(strategy);
        if added_nodes.len() <= removed_nodes.len() {
            return false;
        }
        self.modify(&added_nodes, &removed_nodes);
        // is_valid_after_added_node works if only one node was added
        //todo: must be changed because added_nodes not always contains only one node
        if self.solution_graph.is_node_on_cycle(added_nodes[0]) {
            self.undo_modify(&added_nodes, &removed_nodes);
            self.add_fail();
            return false;
        }
        true
    }
}

#[derive(Clone)]
pub struct AddOneNodeStrategy {}

pub trait UndoableStrategy<G>: Clone {
    fn new() -> Self;
    fn compute<R: Rng>(
        &mut self,
        solution: &UndoableSolution<G, Self>,
        rand: &mut R,
    ) -> (Vec<Node>, Vec<Node>);
}

impl<G> UndoableStrategy<G> for AddOneNodeStrategy
where
    G: GraphEdgeEditing + GraphOrder + Clone + AdjacencyListIn,
{
    fn new() -> Self {
        Self {}
    }

    fn compute<R: Rng>(
        &mut self,
        solution: &UndoableSolution<G, Self>,
        rand: &mut R,
    ) -> (Vec<Node>, Vec<Node>) {
        let nodes_to_add =
            if let Some(&random_node_not_included) = solution.not_included.choose(rand) {
                vec![random_node_not_included]
            } else {
                vec![]
            };

        let nodes_to_delete = vec![];

        (nodes_to_add, nodes_to_delete)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::adj_list_matrix::AdjListMatrixIn;
    use crate::random_models::planted_cycles::generate_planted_cycles;
    use itertools::enumerate;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn create_small_example_graph() -> AdjArrayIn {
        let graph = AdjArrayIn::from(&[
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 5),
            (3, 2),
            (3, 1),
            (3, 4),
        ]);
        graph
    }

    #[test]
    fn abc_on_planted_cycles() {
        let res = generate_planted_cycles(&mut rand::thread_rng(), 150, 3.0, 5, 5, 4);
        let mut graph: AdjListMatrixIn = res.0;
        let planted_cylce_dfvs = res.1;

        println!("Größe des fvs: {}", planted_cylce_dfvs.len());
        let rand = Pcg64::seed_from_u64(123);
        let mut abc: ArtificialBeeColony<_, UndoableSolution<_, AddOneNodeStrategy>, _> =
            ArtificialBeeColony::new(&graph, 5, 10, 5, 5, 10, rand);
        let abc_dfvs = abc.run_abc();
        println!("Berechnetes dfvs: {:?}", abc_dfvs);
        println!("Größe des berechneten dfvs: {:?}", abc_dfvs.len());
        assert!(planted_cylce_dfvs.len() <= abc_dfvs.len());

        graph.remove_edges_of_nodes(&abc_dfvs);
        assert!(graph.is_acyclic());
    }

    #[test]
    fn abc_on_simple_graph() {
        let mut graph = create_small_example_graph();
        let rand = Pcg64::seed_from_u64(123);

        let mut abc: ArtificialBeeColony<_, UndoableSolution<_, AddOneNodeStrategy>, _> =
            ArtificialBeeColony::new(&graph, 5, 10, 5, 5, 10, rand);
        let abc_dfvs = abc.run_abc();
        println!("Berechnetes dfvs: {:?}", abc_dfvs);
        println!("Größe des berechneten dfvs: {:?}", abc_dfvs.len());

        graph.remove_edges_of_nodes(&abc_dfvs);
        assert!(graph.is_acyclic());
    }

    #[test]
    fn test_to_dfvs() {
        let n = 10;
        let dfvs_complement = &vec![0, 1, 5, 6];
        let dfvs = to_dfvs(dfvs_complement, n);
        assert_eq!(dfvs, vec![2, 3, 4, 7, 8, 9])
    }

    #[test]
    fn from_all_nodes_but_fvs() {
        let graph = create_small_example_graph();

        let included_nodes = BitSet::from_slice(graph.len(), &[2, 5]);
        let solution: UndoableSolution<_, AddOneNodeStrategy> =
            UndoableSolution::from_all_nodes_but_fvs(&graph, included_nodes.clone());

        assert_eq!(
            solution.not_included.len(),
            graph.len() - included_nodes.to_vec().len()
        );
        assert_eq!(
            solution.included.to_vec().len(),
            included_nodes.to_vec().len()
        );
        for node in included_nodes.iter() {
            assert!(!solution.not_included.contains(&(node as Node)));
            assert!(solution.included[node]);
        }
    }

    #[test]
    fn try_modify() {
        let graph = create_small_example_graph();
        let in_solution = BitSet::from_slice(graph.len(), &[3, 4, 5]);
        let mut solution: UndoableSolution<_, AddOneNodeStrategy> =
            UndoableSolution::from_all_nodes_but_fvs(&graph, in_solution);
        let rand = &mut Pcg64::seed_from_u64(123);

        assert!(solution.try_modify(rand));
        assert!(!solution.try_modify(rand));
    }

    #[test]
    fn evaluate_fitness() {
        let graph = create_small_example_graph();
        let in_solution = BitSet::from_slice(graph.len(), &[0, 1, 2]);
        let solution: UndoableSolution<_, AddOneNodeStrategy> =
            UndoableSolution::from_all_nodes_but_fvs(&graph, in_solution);

        assert_eq!(solution.evaluate_fitness(), 3);
    }

    #[test]
    fn update_population() {
        let graph = &create_small_example_graph();
        let rand1 = &mut Pcg64::seed_from_u64(123);

        let mut abc: ArtificialBeeColony<_, UndoableSolution<_, AddOneNodeStrategy>, _> =
            ArtificialBeeColony::new(graph, 5, 10, 5, 5, 10, rand1);
        abc._try_keep_min_solutions(2);
        abc.update_population();
        assert_eq!(abc.old_solutions.len(), 0);
        let rand2 = &mut Pcg64::seed_from_u64(123);

        let mut abc2: ArtificialBeeColony<_, UndoableSolution<_, AddOneNodeStrategy>, _> =
            ArtificialBeeColony::new(graph, 0, 2, 4, 2, 2, rand2);
        abc2._try_keep_min_solutions(5);
        for _ in 0..5 {
            let solution =
                BeeSolution::from_all_nodes_but_fvs(graph, BitSet::from_slice(6, &[3, 4, 5]));
            abc2.population.push(solution);
        }
        abc2.best_solution = vec![3, 4, 5];
        abc2.update_population();
        assert_eq!(abc2.old_solutions.len(), 4);
        abc2.update_population();
        abc2.update_population();
        assert_eq!(abc2.population.len(), 4);
        assert_eq!(abc2.old_solutions.len(), 0);
        abc2.update_population();
        abc2.update_population();
        assert_eq!(abc2.population.len(), 0);
    }

    #[test]
    fn add_node() {
        let graph = create_small_example_graph();
        let mut solution: UndoableSolution<_, AddOneNodeStrategy> =
            UndoableSolution::from_all_nodes_but_fvs(&graph, BitSet::new(graph.len()));

        for (i, node) in enumerate(vec![0, 2, 5, 3, 1, 4]) {
            solution.add_node(node);
            assert!(solution.included[node as usize]);
            assert_eq!(solution.included.to_vec().len(), (i + 1) as usize);

            assert!(!solution.not_included.contains(&node));
            assert_eq!(solution.not_included.len(), graph.len() - (i as usize) - 1);
        }
    }

    #[test]
    fn remove_node() {
        let graph = create_small_example_graph();
        let mut solution: UndoableSolution<_, AddOneNodeStrategy> =
            UndoableSolution::from_all_nodes_but_fvs(&graph, BitSet::new_all_set(graph.len()));

        for (i, node) in enumerate(vec![0, 2, 5, 3, 1, 4]) {
            solution.remove_node(node);
            assert!(!solution.included[node as usize]);
            assert_eq!(
                solution.included.to_vec().len(),
                graph.len() - (i as usize) - 1
            );

            assert!(solution.not_included.contains(&node));
            assert_eq!(solution.not_included.len(), (i + 1) as usize);
        }
    }
}
