use crate::graph::Node;
use fxhash::FxHashMap;
use std::collections::hash_map::Entry;

/// A BinaryQueue that allows updating values
pub struct BinaryQueue {
    heap: Vec<Node>,
    values: FxHashMap<Node, i64>,
    indices: FxHashMap<Node, Node>,
}

impl Default for BinaryQueue {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

enum ChildType {
    First,
    Second,
}

impl BinaryQueue {
    /// Creates a new BinaryQueue, allocates **capacity** for each internal data structure.
    /// Guarantees no resizing up-to **capacity** simulatneous elements
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            values: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            indices: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Inserts an **element** with **priority**
    pub fn insert(&mut self, element: Node, priority: i64) {
        match self.values.entry(element) {
            Entry::Occupied(_) => self.update(element, priority),
            Entry::Vacant(entry) => {
                entry.insert(priority);
                self.indices.insert(element, self.heap.len() as Node);
                self.heap.push(element);
                if self.heap.len() > 1 {
                    self.up((self.heap.len() - 1) as Node);
                }
            }
        }
    }

    fn update(&mut self, k: Node, v: i64) {
        *self.values.get_mut(&k).unwrap() = v;
        self.up(*self.indices.get(&k).unwrap());
        self.down(*self.indices.get(&k).unwrap());
    }

    /// Removes and returns the element with the lower priority
    pub fn pop_min(&mut self) -> Option<(Node, i64)> {
        if !self.heap.is_empty() {
            let k = self.heap[0];
            let v = *self.values.get(&k).unwrap();
            self.heap[0] = *self.heap.last().unwrap();
            *self.indices.get_mut(&self.heap[0]).unwrap() = 0;
            self.heap.pop();
            if self.heap.len() > 1 {
                self.down(0);
            }
            return Some((k, v));
        }
        None
    }

    /// Returns the element with the lower priority
    pub fn peek_min(&self) -> Option<(Node, i64)> {
        if !self.heap.is_empty() {
            let k = self.heap[0];
            let v = *self.values.get(&k).unwrap();
            return Some((k, v));
        }
        None
    }

    fn up(&mut self, mut idx: Node) {
        let x = self.heap[idx as usize];
        let mut parent = self.parent(idx);

        loop {
            if parent.is_some()
                && idx > 0
                && self.values.get(&x) < self.values.get(&self.heap[parent.unwrap() as usize])
            {
                let p = parent.unwrap();
                self.heap[idx as usize] = self.heap[p as usize];
                self.indices.insert(self.heap[p as usize], idx);
                idx = p;
                parent = self.parent(idx);
            } else {
                break;
            }
        }
        self.heap[idx as usize] = x;
        self.indices.insert(x, idx);
    }

    fn down(&mut self, idx: Node) {
        let mut current = idx;
        let value = self.heap[current as usize];

        while let Some(mut first) = self.child(current, ChildType::First) {
            if let Some(second) = self.child(current, ChildType::Second) {
                let v1 = self.values.get(&self.heap[second as usize]).unwrap();
                let v2 = self.values.get(&self.heap[first as usize]).unwrap();
                if v1 < v2 {
                    first = second;
                }
            }
            if self.values.get(&self.heap[first as usize]) < self.values.get(&value) {
                self.heap[current as usize] = self.heap[first as usize];
                *self.indices.get_mut(&self.heap[current as usize]).unwrap() = current;
                current = first
            } else {
                break;
            }
        }
        self.heap[current as usize] = value;
        *self.indices.get_mut(&value).unwrap() = current
    }

    fn parent(&self, idx: Node) -> Option<Node> {
        if idx == 0 {
            None
        } else {
            Some((idx - 1) / 2)
        }
    }

    fn child(&self, idx: Node, child_type: ChildType) -> Option<Node> {
        let off = match child_type {
            ChildType::First => 1,
            ChildType::Second => 2,
        };
        let idx = idx * 2 + off;
        if idx >= self.heap.len() as u32 {
            None
        } else {
            Some(idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::binary_queue::BinaryQueue;

    #[test]
    fn pq_pop_min() {
        let mut pq = BinaryQueue::default();

        pq.insert(0, 10);
        pq.insert(16, 1);
        pq.insert(1, 15);

        assert_eq!(pq.pop_min(), Some((16, 1)));
        assert_eq!(pq.pop_min(), Some((0, 10)));
        assert_eq!(pq.pop_min(), Some((1, 15)));
        assert_eq!(pq.pop_min(), None);
    }

    #[test]
    fn pq_update() {
        let mut pq = BinaryQueue::default();

        pq.insert(0, 10);
        pq.insert(16, 1);
        pq.insert(1, 15);
        pq.insert(16, 11);

        assert_eq!(pq.pop_min(), Some((0, 10)));
        assert_eq!(pq.pop_min(), Some((16, 11)));
        assert_eq!(pq.pop_min(), Some((1, 15)));
        assert_eq!(pq.pop_min(), None);
    }
}
