use super::*;
use arrayvec::ArrayVec;
use num::One;
use std::any::Any;
use std::convert::TryFrom;

/// Entry point for the B&B recursion.
///
/// Removes loops (if they exists) as the B&B algorithm  maintains the invariant that each
/// inner node of the recursion tree represents a graph without loops. `upper_limit_excl` is
/// an upper limit of the DFVS size, i.e. if a solution with a cardinality strictly smaller than
/// `upper_limit_excl` exists, the algorithm will return it and `None` otherwise.
pub(super) fn branch_and_bound_impl_start<G: BBGraph + Any>(
    graph: &G,
    upper_limit_excl: Node,
    stats: &mut BBStats,
) -> Option<Solution>
where
    [(); G::CAPACITY]:,
{
    if graph.has_node_with_loop() {
        let (loops_mask, noloops_mask, graph) = graph.remove_loops();
        let num_loops = loops_mask.count_ones();

        if num_loops >= upper_limit_excl {
            return None;
        }

        let mut solution = branch_and_bound_impl_sccs(&graph, upper_limit_excl - num_loops, stats)?;
        solution.shift_from_subgraph(noloops_mask.as_());
        solution.insert_new_nodes(loops_mask.as_());

        Some(solution)
    } else {
        branch_and_bound_impl_sccs(graph, upper_limit_excl, stats)
    }
}

fn branch_and_bound_impl_sccs<G: BBGraph + Any>(
    graph: &G,
    mut upper_limit_excl: Node,
    stats: &mut BBStats,
) -> Option<Solution>
where
    [(); G::CAPACITY]:,
{
    debug_assert!(!graph.has_node_with_loop());

    // at this point we cannot be sure that the graph is not acyclic
    if upper_limit_excl == 0 {
        return None;
    }

    let mut solution = Solution::new();
    let transitive_closure = graph.transitive_closure();
    if !transitive_closure.has_node_with_loop() {
        return Some(solution);
    }

    // since the graph contains at least one cycle, we cannot produce a DFVS with less than one node
    if upper_limit_excl == 1 {
        return None;
    }

    // shortcut if the transitive closure is fully connected; then we do not have to search SCCs
    // and can even avoid repeated computation of the transitive closure
    if transitive_closure.has_all_edges() {
        return branch_and_bound_impl(graph, upper_limit_excl, stats);
    }

    let sccs: ArrayVec<G::NodeMask, { G::CAPACITY }> = transitive_closure
        .sccs()
        .filter(|x| x.count_ones() > 1)
        .collect();

    // each SCC needs at least one node in the DFVS giving a lower bound
    if sccs.len() as Node >= upper_limit_excl {
        return None;
    }
    upper_limit_excl -= sccs.len() as Node;

    for scc in sccs {
        let scc_graph = graph.subgraph(scc);

        let mut scc_solution = branch_and_bound_impl(&scc_graph, upper_limit_excl + 1, stats)?;
        scc_solution.shift_from_subgraph(scc.as_());

        upper_limit_excl -= scc_solution.cardinality() - 1;
        solution.merge(scc_solution);
    }

    Some(solution)
}

fn branch_and_bound_impl<G: BBGraph + Any>(
    graph: &G,
    upper_limit_excl: Node,
    stats: &mut BBStats,
) -> Option<Solution>
where
    [(); G::CAPACITY]:,
{
    debug_assert!(!graph.has_node_with_loop());
    stats.entered_at(graph.len());

    // at this point we cannot be sure that the graph is not acyclic
    if upper_limit_excl == 0 {
        return None;
    }

    if G::CAPACITY == 8 && graph.len() <= 4 {
        if let Some(g8) = (graph as &dyn Any).downcast_ref::<Graph8>() {
            let mut solution = Solution::new();
            solution.insert_new_nodes(Graph4::try_from(g8.clone()).unwrap().lookup() as u64);
            return if solution.cardinality() < upper_limit_excl {
                Some(solution)
            } else {
                None
            };
        }
    } else if G::CAPACITY > 8 && graph.len() <= 8 {
        return branch_and_bound_impl(&Graph8::from_bbgraph(graph), upper_limit_excl, stats);
    } else if G::CAPACITY > 16 && graph.len() <= 16 {
        return branch_and_bound_impl(
            &GenericIntGraph::<u16, 16>::from_bbgraph(graph),
            upper_limit_excl,
            stats,
        );
    } else if G::CAPACITY > 32 && graph.len() <= 32 {
        return branch_and_bound_impl(
            &GenericIntGraph::<u32, 32>::from_bbgraph(graph),
            upper_limit_excl,
            stats,
        );
    }

    let solution1 = {
        let subgraph = graph.remove_first_node();
        let subgraph_mask = graph.nodes_mask() - G::NodeMask::one();

        branch_and_bound_impl_sccs(&subgraph, upper_limit_excl - 1, stats).map(|mut sol| {
            sol.shift_from_subgraph(subgraph_mask.as_())
                .insert_new_node(0);
            sol
        })
    };

    if graph.first_node_has_loop() || solution1.as_ref().map_or(false, |x| x.cardinality() == 0) {
        return solution1;
    }

    let upper_limit_excl = solution1
        .as_ref()
        .map_or(upper_limit_excl, |x| x.cardinality());

    let solution2 = {
        let (loops_mask, noloops_mask, subgraph) = graph.contract_first_node().remove_loops();

        if loops_mask.count_ones() >= upper_limit_excl {
            // This can happen if solution1 is already better than the current solution
            // plus the self-loops from contracting.
            None
        } else {
            let sol =
                branch_and_bound_impl(&subgraph, upper_limit_excl - loops_mask.count_ones(), stats);
            if let Some(mut sol) = sol {
                sol.shift_from_subgraph((noloops_mask << 1).as_())
                    .insert_new_nodes((loops_mask << 1).as_());
                Some(sol)
            } else {
                None
            }
        }
    };

    // if solution 2 found a DFVS it is smaller than `upper_limit_excl` and therefore strictly
    // smaller than solution 1 (if it exists)
    if solution2.is_some() {
        solution2
    } else {
        solution1
    }
}
