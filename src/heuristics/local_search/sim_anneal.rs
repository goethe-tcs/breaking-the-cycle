use super::topo::topo_config::{TopoConfig, TopoGraph, TopoMoveStrategy};
use super::topo::topo_local_search::TopoLocalSearch;
use crate::algorithm::{IterativeAlgorithm, TerminatingIterativeAlgorithm};
use crate::bench::io::keyed_buffer::KeyedBuffer;
use crate::graph::Node;
use rand::Rng;

/// Implementation of the simulated annealing algorithm that is presented in the
/// "Applying local search to the feedback vertex set problem" paper by Philippe Galinier et al.
pub struct SimAnneal<'a, T, S, G, R> {
    local_search: TopoLocalSearch<T, S, G>,
    rng: &'a mut R,
    /// Best DFVS that was found so far
    best_fvs: Vec<Node>,
    /// The current temperature. Dictates the probability of performing a bad move, which would
    /// increase the size of the DFVS
    curr_temp: f64,

    // fixed algo parameters
    /// Amount of moves that are evaluated in every stage
    max_stage_evals: usize,
    /// Amount of stages that have to fail in succession in order for this algorithm to fail
    max_stage_fails: usize,
    /// Used to decrease the temperature after each stage
    temp_reduce_fac: f64,

    // stop condition related fields
    /// Amount of stages that were unsuccessful in succession
    failed_stages: usize,
    /// Amount of moves that were evaluated in the current stage
    evals_this_stage: usize,
    /// Whether the size of the DFVS was decreased in the current stage so far
    was_stage_successful: bool,
    /// Whether the local search has no more moves to offer
    is_local_search_completed: bool,

    // fields for analyzing algo
    move_evals_total: usize,
    moves_total: usize,
}

impl<'a, T, S, G, R> SimAnneal<'a, T, S, G, R>
where
    T: TopoConfig<'a, G> + 'a,
    S: TopoMoveStrategy<G>,
    G: TopoGraph,
    R: Rng,
{
    pub fn new(
        local_search: TopoLocalSearch<T, S, G>,
        max_stage_evals: usize,
        max_stage_fails: usize,
        start_temperature: f64,
        temp_reduce_fac: f64,
        rng: &'a mut R,
    ) -> Self {
        let best_fvs = local_search.fvs().to_vec();

        Self {
            max_stage_evals,
            max_stage_fails,
            temp_reduce_fac,
            local_search,
            rng,
            best_fvs,
            curr_temp: start_temperature,
            failed_stages: 0,
            evals_this_stage: 0,
            was_stage_successful: false,
            is_local_search_completed: false,
            move_evals_total: 0,
            moves_total: 0,
        }
    }

    pub fn write_metrics(&self, writer: &mut KeyedBuffer) {
        writer.write("move_evals_total", self.move_evals_total);
        writer.write("moves_total", self.moves_total);
    }

    fn next_stage(&mut self) {
        if self.was_stage_successful {
            self.failed_stages = 0;
        } else {
            self.failed_stages += 1;
        }

        self.evals_this_stage = 0;
        self.was_stage_successful = false;
        self.curr_temp *= self.temp_reduce_fac;
    }
}

impl<'a, T, S, G, R> IterativeAlgorithm for SimAnneal<'a, T, S, G, R>
where
    T: TopoConfig<'a, G> + 'a,
    S: TopoMoveStrategy<G>,
    G: TopoGraph,
    R: Rng,
{
    fn execute_step(&mut self) {
        // get next move
        let next_move = self.local_search.next_move();
        if next_move.is_none() {
            self.is_local_search_completed = true;
            return;
        }
        let next_move = next_move.unwrap();
        let delta = next_move.performance();
        self.evals_this_stage += 1;
        self.move_evals_total += 1;

        // decide whether to execute the move
        if delta <= 0 || self.rng.gen_bool((-delta as f64 / self.curr_temp).exp()) {
            self.local_search.perform_move(next_move);
            self.moves_total += 1;
            if self.local_search.fvs().len() < self.best_fvs.len() {
                self.best_fvs = self.local_search.fvs().to_vec();
                self.was_stage_successful = true;
            }
        } else {
            self.local_search.reject_move(next_move);
        }

        // move to next stage
        if self.evals_this_stage >= self.max_stage_evals {
            self.next_stage();
        }
    }

    fn is_completed(&self) -> bool {
        self.failed_stages >= self.max_stage_fails || self.is_local_search_completed
    }

    fn best_known_solution(&mut self) -> Option<&[Node]> {
        Some(&self.best_fvs)
    }
}

impl<'a, T, S, G, R> TerminatingIterativeAlgorithm for SimAnneal<'a, T, S, G, R>
where
    T: TopoConfig<'a, G> + 'a,
    S: TopoMoveStrategy<G>,
    G: TopoGraph,
    R: Rng,
{
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bench::fvs_bench::test_utils::test_algo_with_pace_graphs;
    use crate::graph::adj_array::AdjArrayIn;
    use crate::graph::{GraphNew, Traversal};
    use crate::heuristics::local_search::topo::rand_topo_strategy::RandomTopoStrategy;
    use crate::heuristics::local_search::topo::vec_topo_config::VecTopoConfig;
    use crate::heuristics::utils::apply_fvs_to_graph;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn test_simple_cyclic_graph() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            RandomTopoStrategy::new(&mut strategy_rng, 7),
        );
        let mut sim_anneal = SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
        let fvs = sim_anneal.run_to_completion().unwrap();
        assert_eq!(fvs.len(), 1);

        let reduced_graph = apply_fvs_to_graph(&graph, fvs);
        assert!(reduced_graph.is_acyclic());
    }

    #[test]
    fn test_with_graphs() {
        test_algo_with_pace_graphs("Simulated Annealing", |graph, _, _, _| {
            let mut strategy_rng = Pcg64::seed_from_u64(0);
            let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
            let topo_config = VecTopoConfig::new(&graph);
            let strategy = RandomTopoStrategy::new(&mut strategy_rng, 7);
            let local_search = TopoLocalSearch::new(topo_config, strategy);
            let mut sim_anneal =
                SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);
            sim_anneal.run_to_completion().unwrap()
        })
        .unwrap();
    }

    #[test]
    fn is_local_search_completed() {
        let graph = AdjArrayIn::new(1);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            RandomTopoStrategy::new(&mut strategy_rng, 7),
        );
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let mut sim_anneal = SimAnneal::new(local_search, 20, 20, 1.0, 0.9, &mut sim_anneal_rng);

        sim_anneal.execute_step();
        assert!(!sim_anneal.is_local_search_completed);

        sim_anneal.execute_step();
        assert!(sim_anneal.is_local_search_completed);
        assert!(sim_anneal.is_completed());
    }

    #[test]
    fn stop_condition() {
        let graph = AdjArrayIn::from(&[(0, 1), (1, 0)]);

        let mut strategy_rng = Pcg64::seed_from_u64(0);
        let local_search = TopoLocalSearch::new(
            VecTopoConfig::new(&graph),
            RandomTopoStrategy::new(&mut strategy_rng, 7),
        );
        let mut sim_anneal_rng = Pcg64::seed_from_u64(1);
        let mut sim_anneal = SimAnneal::new(local_search, 2, 3, 1.0, 0.9, &mut sim_anneal_rng);
        assert_eq!(sim_anneal.evals_this_stage, 0);
        assert_eq!(sim_anneal.failed_stages, 0);
        assert_eq!(sim_anneal.failed_stages, 0);

        sim_anneal.execute_step();
        assert_eq!(sim_anneal.evals_this_stage, 1);

        sim_anneal.execute_step();
        assert_eq!(sim_anneal.failed_stages, 0);
        assert_eq!(sim_anneal.evals_this_stage, 0);

        sim_anneal.execute_step();
        sim_anneal.execute_step();
        assert_eq!(sim_anneal.failed_stages, 1);
        assert_eq!(sim_anneal.evals_this_stage, 0);

        sim_anneal.run_to_completion();
        assert_eq!(sim_anneal.move_evals_total, 8);
    }
}
