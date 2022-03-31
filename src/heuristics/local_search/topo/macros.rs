// cargo check does not detect imports and macro use in test modules. This is the only workaround
#![allow(unused_macros, unused_imports)]

macro_rules! keys {
    ($priority_queue:expr) => {
        $priority_queue.iter().map(|(key, _)| key).copied()
    };
}

macro_rules! priorities {
    ($priority_queue:expr) => {
        $priority_queue.iter().map(|(_, prio)| prio).copied()
    };
}

macro_rules! topo_strat_base_tests {
    ($factory_func:ident) => {};
}

macro_rules! topo_config_base_tests {
    ($factory_func:ident) => {
        use crate::graph::adj_array::AdjArrayIn;
        use crate::heuristics::local_search::topo::topo_config::MovePosition;

        #[test]
        fn set_state() {
            let graph = AdjArrayIn::from(&[(0, 1), (0, 2), (0, 3), (0, 4)]);
            let mut topo_config = $factory_func(&graph);

            let topo_order = vec![0, 3, 2];
            let fvs = vec![1, 4];
            topo_config.set_state(topo_order.clone(), fvs.clone());

            assert_eq!(topo_config.as_slice(), topo_order);
            assert_eq!(topo_config.fvs(), fvs);
        }

        #[test]
        fn set_state_from_fvs_basic() {
            let graph = AdjArrayIn::from(&[(0, 1), (0, 2), (0, 3), (0, 4)]);
            let mut topo_config = $factory_func(&graph);

            let fvs = vec![4, 2];
            topo_config.set_state_from_fvs(fvs.clone());

            assert_eq!(topo_config.as_slice(), &[0, 3, 1]);
            assert_eq!(topo_config.fvs(), fvs);
        }

        #[test]
        fn set_state_from_fvs_small_circle() {
            let graph = AdjArrayIn::from(&[(0, 1), (1, 2), (2, 3), (3, 4)]);
            let mut topo_config = $factory_func(&graph);

            let fvs = vec![1];
            topo_config.set_state_from_fvs(fvs.clone());

            assert_eq!(topo_config.as_slice(), &[2, 3, 4, 0]);
            assert_eq!(topo_config.fvs(), fvs);
        }

        #[test]
        fn test_get_index_of_node() {
            let graph = AdjArrayIn::from(&[(0, 1), (0, 2), (0, 3), (0, 4)]);
            let mut topo_config = $factory_func(&graph);

            for node in 0..5 {
                let topo_move = topo_config.create_move(node, node as usize);
                topo_config.perform_move(topo_move);

                assert_eq!(node as usize, topo_config.get_index(&node).unwrap());
            }
        }

        #[test]
        fn test_calc_conflicts() {
            let graph = AdjArrayIn::from(&[(0, 1), (1, 0)]);
            let mut topo_config = $factory_func(&graph);

            let topo_move = topo_config.create_move(0, 0);
            topo_config.perform_move(topo_move);

            let conflicting_neighbors: Vec<_> =
                topo_config.calc_conflicts(1, MovePosition::Index(0));
            assert_eq!(vec![(0, 0)], conflicting_neighbors);
        }

        #[test]
        fn test_calc_move_candidates() {
            let graph = AdjArrayIn::from(&[(0, 1), (2, 0), (0, 3), (4, 0), (5, 0), (0, 6), (0, 7)]);
            let topo_order = vec![1, 2, 3, 4, 5, 6, 7];
            let fvs = vec![0];
            let mut topo_config = $factory_func(&graph);
            topo_config.set_state(topo_order.clone(), fvs.clone());

            let (i_minus_move, i_plus_move) = topo_config.calc_move_candidates(0);
            assert_eq!(i_minus_move.position(), MovePosition::Index(5));
            assert_eq!(i_plus_move.position(), MovePosition::Index(0));

            let i_minus_conflicts_expected = vec![(1, 0), (3, 2)];
            let mut i_minus_conflicts = i_minus_move.conflicting_nodes().unwrap().clone();
            i_minus_conflicts.sort();
            assert_eq!(i_minus_conflicts, i_minus_conflicts_expected);
            let mut i_minus_conflicts = topo_config.calc_conflicts(0, MovePosition::Index(5));
            i_minus_conflicts.sort();
            assert_eq!(i_minus_conflicts, i_minus_conflicts_expected);

            let i_plus_conflicts_expected = vec![(2, 1), (4, 3), (5, 4)];
            let mut i_plus_conflicts = i_plus_move.conflicting_nodes().unwrap().clone();
            i_plus_conflicts.sort();
            assert_eq!(i_plus_conflicts, i_plus_conflicts_expected);
            let mut i_plus_conflicts = topo_config.calc_conflicts(0, MovePosition::Index(0));
            i_plus_conflicts.sort();
            assert_eq!(i_plus_conflicts, i_plus_conflicts_expected);
        }

        #[test]
        fn test_calc_move_candidates_2() {
            let graph = AdjArrayIn::from(&[(1, 0), (2, 0), (0, 3), (4, 0), (5, 0), (0, 6), (0, 7)]);
            let topo_order = vec![1, 2, 3, 4, 5, 6, 7];
            let fvs = vec![0];
            let mut topo_config = $factory_func(&graph);
            topo_config.set_state(topo_order.clone(), fvs.clone());

            let (i_minus_move, i_plus_move) = topo_config.calc_move_candidates(0);
            assert_eq!(i_minus_move.position(), MovePosition::Index(5));
            assert_eq!(i_plus_move.position(), MovePosition::Index(2));

            let i_minus_conflicts_expected = vec![(3, 2)];
            let mut i_minus_conflicts = i_minus_move.conflicting_nodes().unwrap().clone();
            i_minus_conflicts.sort();
            assert_eq!(i_minus_conflicts, i_minus_conflicts_expected);
            let mut i_minus_conflicts = topo_config.calc_conflicts(0, MovePosition::Index(5));
            i_minus_conflicts.sort();
            assert_eq!(i_minus_conflicts, i_minus_conflicts_expected);

            let i_plus_conflicts_expected = vec![(4, 3), (5, 4)];
            let mut i_plus_conflicts = i_plus_move.conflicting_nodes().unwrap().clone();
            i_plus_conflicts.sort();
            assert_eq!(i_plus_conflicts, i_plus_conflicts_expected);
            let mut i_plus_conflicts = topo_config.calc_conflicts(0, MovePosition::Index(2));
            i_plus_conflicts.sort();
            assert_eq!(i_plus_conflicts, i_plus_conflicts_expected);
        }

        #[test]
        fn test_perform_move() {
            let graph = AdjArrayIn::from(&[(0, 1), (1, 0), (2, 1)]);
            let mut topo_config = $factory_func(&graph);

            let mut topo_move = topo_config.create_move(0, 0);
            topo_config.perform_move(topo_move);
            assert_eq!(topo_config.as_slice(), &[0]);

            topo_move = topo_config.create_move(1, 0);
            topo_config.perform_move(topo_move);
            assert_eq!(topo_config.as_slice(), &[1]);

            topo_move = topo_config.create_move(2, 1);
            topo_config.perform_move(topo_move);
            assert_eq!(topo_config.as_slice(), &[2]);

            topo_move = topo_config.create_move(1, 1);
            topo_config.perform_move(topo_move);
            assert_eq!(topo_config.as_slice(), &[2, 1]);
        }
    };
}

pub(in crate::heuristics::local_search) use keys;
pub(in crate::heuristics::local_search) use priorities;
pub(in crate::heuristics::local_search) use topo_config_base_tests;
