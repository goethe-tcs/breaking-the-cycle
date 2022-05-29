use super::*;
use fxhash::{FxBuildHasher, FxHashMap};

const INITIAL_CAPACITY: usize = 10_000;
const DEFAULT_MAX_CAPACITY: usize = 20_000_000; // roughly translates to 1 GB
const EVICITION_SEACH: usize = 10;

pub struct ResultCache {
    cache: FxHashMap<String, (u64, Node, OptSolution)>,
    capacity: usize,
    timestamp: u64,
    number_of_misses: u64,
    number_of_accesses: u64,
}

impl Default for ResultCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultCache {
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::with_capacity_and_hasher(INITIAL_CAPACITY, FxBuildHasher::default()),
            capacity: DEFAULT_MAX_CAPACITY,
            timestamp: 0,
            number_of_misses: 0,
            number_of_accesses: 0,
        }
    }

    /// Sets new capacity of the cache without reserving actual memory.
    ///
    /// # Warning
    /// Erases cache if the current number of elements exceeds the new cache size
    pub fn set_capacity(&mut self, capacity: usize) {
        if self.cache.len() > capacity {
            self.cache = FxHashMap::with_capacity_and_hasher(capacity + 1, FxBuildHasher::default())
        }
        self.capacity = capacity;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn number_of_cache_hits(&self) -> u64 {
        self.number_of_accesses - self.number_of_misses
    }

    pub fn number_of_cache_misses(&self) -> u64 {
        self.number_of_misses
    }

    /// Introduces new result into cache
    pub fn add_to_cache(&mut self, digest: String, result: OptSolution, upper_bound: Node) {
        if self.cache.len() > self.capacity {
            self.evict_element()
        }
        self.cache
            .insert(digest, (self.timestamp, upper_bound, result));
        self.timestamp += 1;
    }

    /// Looks up result from cache. We need to pass the upper_bound, since a previous entry
    /// obtained for a lower upper bound might be `None` and apply anymore.
    pub fn get(&mut self, digest: &String, upper_bound: Node) -> Option<OptSolution> {
        self.number_of_accesses += 1;
        if let Some((timestamp, ub, result)) = self.cache.get_mut(digest) {
            *timestamp = self.timestamp;

            if let Some(result) = result {
                if upper_bound as usize > result.len() {
                    return Some(Some(result.clone()));
                } else {
                    return Some(None);
                }
            } else if *ub >= upper_bound {
                return Some(None);
            }
        }
        self.number_of_misses += 1;
        None
    }

    fn evict_element(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        // we assume that the iteration order of the hash table is random
        // then we do not have a considerable bias from only evicting from the beginning
        let key_to_evict = self
            .cache
            .iter()
            .take(EVICITION_SEACH)
            .min_by_key(|(_, (t, _, _))| *t)
            .unwrap()
            .0
            .clone();
        self.cache.remove(&key_to_evict);
    }
}
