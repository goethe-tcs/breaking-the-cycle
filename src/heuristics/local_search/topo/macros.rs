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

macro_rules! impl_helper_topo_new_with_fvs {
    () => {
        pub fn new_with_fvs<I>(graph: &'a G, fvs: I) -> Self
        where
            I: IntoIterator<Item = Node> + Clone,
        {
            let mut result = Self::new(graph);
            result.set_state_from_fvs(fvs);
            result
        }
    };
}

macro_rules! topo_config_base_tests {
    ($factory_func:ident) => {
        use crate::graph::adj_array::AdjArrayIn;

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
        fn test_get_conflicting_neighbors() {
            let graph = AdjArrayIn::from(&[(0, 1), (1, 0)]);
            let mut topo_config = $factory_func(&graph);

            let topo_move = topo_config.create_move(0, 0);
            topo_config.perform_move(topo_move);

            let conflicting_neighbors: Vec<_> = topo_config.get_conflicting_neighbors(1, 0);
            assert_eq!(vec![(0, 0)], conflicting_neighbors);
        }

        #[test]
        fn test_perform_move() {
            let graph = AdjArrayIn::from(&[(0, 1), (1, 0)]);
            let mut topo_config = $factory_func(&graph);

            let mut topo_move = topo_config.create_move(0, 0);
            topo_config.perform_move(topo_move);

            topo_move = topo_config.create_move(1, 0);
            topo_config.perform_move(topo_move);

            assert_eq!(vec![1], topo_config.as_slice().to_vec());
        }
    };
}

pub(in crate::heuristics::local_search) use impl_helper_topo_new_with_fvs;
pub(in crate::heuristics::local_search) use keys;
pub(in crate::heuristics::local_search) use priorities;
pub(in crate::heuristics::local_search) use topo_config_base_tests;
