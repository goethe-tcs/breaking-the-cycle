use super::*;
use arrayvec::ArrayVec;
use num::{One, Zero};
use std::any::Any;

/// This is the entry point for the branch and bound recursion. It support arbitrary topologies,
/// applies some reductions and then recurses on each strongly connected component.
pub fn branch_and_bound_impl_sccs<G>(
    graph: &G,
    mut upper_limit_excl: Node,
    stats: &mut BBStats,
) -> Option<G::NodeMask>
where
    G: BBGraph
        + Any
        + BBTryCompact<Graph8>
        + BBTryCompact<Graph16>
        + BBTryCompact<Graph32>
        + BBTryCompact<Graph64>,
    [(); G::CAPACITY]:,
{
    // at this point we cannot be sure that the graph is not acyclic
    if upper_limit_excl == 0 {
        return None;
    }

    // contract chaining nodes and remove loops
    let graph = graph.contract_chaining_nodes();
    let loops = graph.nodes_with_loops();
    if loops.count_ones() >= upper_limit_excl {
        return None;
    }
    upper_limit_excl -= loops.count_ones();
    let graph = graph.remove_edges_at_nodes(loops);

    let transitive_closure = graph.transitive_closure();
    if !transitive_closure.has_node_with_loop() {
        return Some(loops);
    }

    // since the graph contains at least one cycle, we cannot produce a DFVS with less than one node
    if upper_limit_excl == 1 {
        return None;
    }

    // shortcut if the transitive closure is fully connected; then we do not have to search SCCs
    // and can even avoid repeated computation of the transitive closure
    if loops.is_zero() && transitive_closure.has_all_edges() {
        return branch_and_bound_impl(&graph, upper_limit_excl, stats);
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

    let mut solution = loops;
    for scc in sccs {
        let scc_graph = graph.subgraph(scc);

        let scc_solution = branch_and_bound_impl(&scc_graph, upper_limit_excl + 1, stats)?;

        upper_limit_excl -= scc_solution.count_ones() - 1;
        solution |= scc_solution.bit_deposit(scc);
    }

    Some(solution)
}

macro_rules! return_if_some {
    ($e:expr) => {
        if let Some(x) = { $e } {
            return x;
        }
    };
}

fn branch_and_bound_impl<G>(graph: &G, ulimit_ex: Node, stats: &mut BBStats) -> Option<G::NodeMask>
where
    G: BBGraph
        + Any
        + BBTryCompact<Graph8>
        + BBTryCompact<Graph16>
        + BBTryCompact<Graph32>
        + BBTryCompact<Graph64>,
    [(); G::CAPACITY]:,
{
    debug_assert!(!graph.has_node_with_loop());
    stats.entered_at(graph.len());

    // at this point we cannot be sure that the graph is not acyclic, so we have some annoying checks
    if ulimit_ex == 0 {
        return None;
    }

    if graph.len() < 2 {
        return if graph.len() == 0 {
            Some(G::NodeMask::zero())
        } else if graph.first_node_has_loop() {
            if ulimit_ex == 1 {
                None
            } else {
                Some(G::NodeMask::one())
            }
        } else {
            Some(G::NodeMask::zero())
        };
    }

    // We try to switch to a smaller graph representation; the compiler will remove all branches
    // to representations larger than the current one ... hopefully ;)
    return_if_some!(graph.try_compact(|g: &Graph8| branch_and_bound_impl(g, ulimit_ex, stats)));
    return_if_some!(graph.try_compact(|g: &Graph16| branch_and_bound_impl(g, ulimit_ex, stats)));
    return_if_some!(graph.try_compact(|g: &Graph32| branch_and_bound_impl(g, ulimit_ex, stats)));

    // Now to the actual branching. In general there are two options
    // - branch1: Delete node 0
    // - branch2: Contract node 0

    // To speed things up, we try to choose a good node and make it node 0
    let node_to_process = graph
        .node_with_most_undirected_edges()
        .or_else(|| graph.node_with_max_out_degree())
        .unwrap_or(0);
    let graph = graph.swap_nodes(0, node_to_process);

    let solution1 = {
        if graph.is_chain_node(0) {
            // chain nodes do not have to be deleted; skip them and contract them in branch 2
            None
        } else {
            branch_and_bound_impl_sccs(&graph.remove_first_node(), ulimit_ex - 1, stats)
                .map(|sol| (sol << 1) | G::NodeMask::one())
        }
    };

    if graph.first_node_has_loop() {
        // a loop cannot be contracted an solution1 is a good as it gets
        return solution1;
    }

    let ulimit_ex = solution1.as_ref().map_or(ulimit_ex, |x| x.count_ones());

    let solution2 =
        branch_and_bound_impl_sccs(&graph.contract_first_node(), ulimit_ex, stats).map(|s| s << 1);

    // if solution 2 found a DFVS it is smaller than `ulimit_ex` and therefore strictly
    // smaller than solution 1 (if it exists)
    let best_solution = if solution2.is_some() {
        solution2
    } else {
        solution1
    };

    best_solution.map(|sol| sol.exchange_bits(0, node_to_process as usize))
}
