// cargo check does not detect imports and macro use in test modules. This is the only workaround
#![allow(unused_macros, unused_imports)]

macro_rules! set_tests {
    ($set_type:ty, $factory_func:ident) => {
        use itertools::Itertools;
        use rand::SeedableRng;
        use rand_pcg::Pcg64;
        use std::collections::HashSet;
        use test_case::test_case;

        fn assert_set_vec_eq(actual: $set_type, expected: Vec<Node>) {
            assert_eq!(actual.vec, expected);
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.is_empty(), expected.is_empty());
            assert_eq!(actual.as_slice(), expected.as_slice());
            assert_eq!(actual.iter().collect_vec(), expected.iter().collect_vec());
            assert_eq!(
                actual.clone().iter_mut().collect_vec(),
                expected.clone().iter_mut().collect_vec()
            );

            for (i, element) in expected.into_iter().enumerate() {
                assert!(actual.contains(&element));
                assert_eq!(actual.get_index(&element).unwrap(), i);
            }
        }

        #[test_case(&[], 2, &[2])]
        #[test_case(&[5], 5, &[5])]
        #[test_case(&[6], 1, &[6, 1])]
        #[test_case(&[4, 3], 4, &[4, 3])]
        #[test_case(&[5, 2], 6, &[5, 2, 6])]
        #[test_case(&[6, 4, 3], 4, &[6, 4, 3])]
        fn insert(input: &[Node], node: Node, expected: &[Node]) {
            let mut set = $factory_func(input.to_vec(), 10);
            assert_eq!(!set.contains(&node), set.insert(node));

            assert_set_vec_eq(set, expected.to_vec());
        }

        #[test_case(&[], 0, 5, &[5])]
        #[test_case(&[1, 3], 1, 2, &[1, 2, 3])]
        #[test_case(&[1, 2], 2, 3, &[1, 2, 3])]
        #[test_case(&[1, 2, 4, 5], 2, 3, &[1, 2, 3, 4, 5])]
        #[test_case(&[1, 2, 4, 5], 0, 6, &[6, 1, 2, 4, 5])]
        #[test_case(&[1, 2, 4, 6], 0, 6, &[6, 1, 2, 4])]
        #[test_case(&[3, 2, 4, 6], 3, 3, &[2, 4, 6, 3])]
        #[test_case(&[3, 7, 4, 6], 1, 4, &[3, 4, 7, 6])]
        fn insert_at(input: &[Node], index: usize, node: Node, expected: &[Node]) {
            let mut set = $factory_func(input.to_vec(), 8);
            set.insert_at(index, node);

            assert_set_vec_eq(set, expected.to_vec());
        }

        #[test_case(&[3], 3, &[])]
        #[test_case(&[3], 5, &[3])]
        #[test_case(&[7, 2, 9], 7, &[9, 2])]
        #[test_case(&[4, 1, 8, 5], 1, &[4, 5, 8])]
        #[test_case(&[3, 5, 7, 2], 2, &[3, 5, 7])]
        fn swap_remove(input: &[Node], node: Node, expected: &[Node]) {
            let mut set = $factory_func(input.to_vec(), 13);
            assert_eq!(set.contains(&node), set.swap_remove(&node).is_some());

            assert_set_vec_eq(set, expected.to_vec());
        }

        #[test_case(&[], 3, &[])]
        #[test_case(&[3], 3, &[])]
        #[test_case(&[7, 2, 9], 7, &[2, 9])]
        #[test_case(&[4, 1, 8, 5], 1, &[4, 8, 5])]
        #[test_case(&[3, 5, 7, 2], 2, &[3, 5, 7])]
        #[test_case(&[3, 5, 7, 2], 9, &[3, 5, 7, 2])]
        fn shift_remove(input: &[Node], node: Node, expected: &[Node]) {
            let mut set = $factory_func(input.to_vec(), 20);
            assert_eq!(set.contains(&node), set.shift_remove(&node).is_some());

            assert_set_vec_eq(set, expected.to_vec());
        }

        #[test_case(&[3], &[3], &[])]
        #[test_case(&[3], &[3, 2], &[])]
        #[test_case(&[7, 2, 9], &[7], &[2, 9])]
        #[test_case(&[4, 1, 8, 5], &[1], &[4, 8, 5])]
        #[test_case(&[3, 5, 7, 2], &[2], &[3, 5, 7])]
        #[test_case(&[7, 2, 9, 5], &[7, 9], &[2, 5])]
        #[test_case(&[3, 5, 7, 2], &[7, 2], &[3, 5])]
        #[test_case(&[2, 6, 1], &[2, 6], &[1])]
        #[test_case(&[2, 6, 1, 7], &[7, 2, 6, 1], &[])]
        #[test_case(&[4, 1, 8, 5, 3], &[1, 4], &[8, 5, 3])]
        fn shift_remove_bulk(input: &[Node], nodes: &[Node], expected: &[Node]) {
            let mut set = $factory_func(input.to_vec(), 11);
            set.shift_remove_bulk(nodes.iter());

            assert_set_vec_eq(set, expected.to_vec());
        }

        #[test_case(&[0], 2 => (vec![Some(0), None], vec![]))]
        #[test_case(&[0, 1, 2], 1 => (vec![Some(2)], vec![0, 1]))]
        #[test_case(&[1, 0, 2], 4 => (vec![Some(2), Some(0), Some(1), None], vec![]))]
        fn pop(input: &[Node], pop_amount: usize) -> (Vec<Option<u32>>, Vec<u32>) {
            let mut index_set = $factory_func(input.to_vec(), 20);

            let mut popped_nodes = vec![];
            for _ in 0..pop_amount {
                popped_nodes.push(index_set.pop());
            }

            (popped_nodes, index_set.as_slice().to_vec())
        }

        #[test_case(&[], 1)]
        #[test_case(&[3], 1)]
        #[test_case(&[7, 1], 1000)]
        #[test_case(&[9, 4, 6], 1000)]
        fn choose(input: &[Node], sample_count: usize) {
            let set = $factory_func(input.to_vec(), 10);
            let mut rng = Pcg64::seed_from_u64(0);

            let mut samples = HashSet::with_capacity(input.len());
            for _ in 0..sample_count {
                if let Some(node) = set.choose(&mut rng) {
                    samples.insert(*node);
                }
            }

            let actual_samples = samples.iter().sorted().collect_vec();
            let expected_samples = input.iter().sorted().collect_vec();
            assert_eq!(actual_samples, expected_samples);
        }
    };
}

pub(in crate::heuristics) use set_tests;
