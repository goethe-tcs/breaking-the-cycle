use super::*;
use crate::bitset::BitSet;

pub struct UniqueNodeStack {
    stack: Vec<Node>,
    bitset: BitSet,
}

impl UniqueNodeStack {
    pub fn new(n: usize) -> Self {
        Self {
            stack: Vec::with_capacity(n),
            bitset: BitSet::new(n),
        }
    }

    /// Returns the element that would be popped if [`UniqueNodeStack::pop`] were called but does
    /// not alter the data structure
    pub fn peek(&self) -> Option<Node> {
        Some(*self.stack.last()?)
    }

    /// Removes last element pushes onto stack (and None if none such element exists)
    pub fn pop(&mut self) -> Option<Node> {
        let u = self.stack.pop()?;
        self.bitset.unset_bit(u as usize);
        Some(u)
    }

    /// Tries to push new element onto stack if it is not currently stored on the stack.
    /// Returns true iff the push succeeded
    pub fn try_push(&mut self, u: Node) -> bool {
        if self.bitset.set_bit(u as usize) {
            return false;
        }

        self.stack.push(u);
        true
    }

    /// Executes try_push on every element of the iterator provided and returns the number of
    /// successful insertions.
    pub fn try_push_iter<T: IntoIterator<Item = Node>>(&mut self, iter: T) -> usize {
        iter.into_iter().map(|u| self.try_push(u) as usize).sum()
    }

    /// Returns the number of elements on the stack
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Returns true if currently no elements are stored on stack
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Return true if the requested element is currently stored on the stack
    pub fn contains(&self, u: Node) -> bool {
        self.bitset[u as usize]
    }
}

#[cfg(test)]
mod test {
    use super::UniqueNodeStack;

    #[test]
    fn test_unique_node_stack() {
        let mut stack = UniqueNodeStack::new(10);

        let assert_len = |stack: &UniqueNodeStack, n| {
            assert_eq!(stack.is_empty(), n == 0);
            assert_eq!(stack.len(), n);
        };

        assert_eq!(stack.contains(3), false);
        assert_len(&stack, 0);
        assert!(stack.pop().is_none());
        assert!(stack.peek().is_none());
        assert_len(&stack, 0);
        assert!(stack.try_push(5));
        assert_eq!(stack.peek(), Some(5));
        assert_eq!(stack.contains(3), false);
        assert_len(&stack, 1);
        assert!(!stack.try_push(5));
        assert_len(&stack, 1);
        assert!(stack.try_push(3));
        assert_eq!(stack.peek(), Some(3));
        assert_eq!(stack.contains(3), true);
        assert_len(&stack, 2);
        assert_eq!(stack.try_push_iter([1, 3, 5, 6]), 2);
        assert_len(&stack, 4);
        assert_eq!(stack.pop(), Some(6));
        assert_len(&stack, 3);
        assert_eq!(stack.pop(), Some(1));
        assert_len(&stack, 2);
        assert_eq!(stack.contains(3), true);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.contains(3), false);
        assert_len(&stack, 1);
        assert_eq!(stack.pop(), Some(5));
        assert_len(&stack, 0);
        assert!(stack.pop().is_none());
        assert_len(&stack, 0);
    }
}
