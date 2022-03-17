#![doc = include_str!("../README.md")]
#![allow(clippy::comparison_chain)]
#![allow(clippy::len_without_is_empty)]

mod base_node;
mod key;
mod lock;
mod node_16;
mod node_256;
mod node_4;
mod node_48;
mod node_ptr;
mod tree;
mod utils;

mod node_lock;

mod range_scan;

#[cfg(feature = "stats")]
mod stats;

#[cfg(test)]
mod tests;

use std::{mem::ManuallyDrop, vec};

use base_node::BaseNode;
use key::RawKey;
use key::UsizeKey;
use tree::RawTree;

/// Types needed to safely access shared data concurrently.
pub mod epoch {
    pub use crossbeam_epoch::{pin, Guard};
}

/// ArtUsize is a special case for [Art] where the key is a usize.
/// It can have better performance
pub struct ArtUsize {
    inner: RawTree<UsizeKey>,
}

impl Default for ArtUsize {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtUsize {
    /// Returns a copy of the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), 42);
    /// ```
    #[inline]
    pub fn get(&self, key: &usize, guard: &epoch::Guard) -> Option<usize> {
        let key = UsizeKey::key_from(*key);
        self.inner.get(&key, guard)
    }

    /// Enters an epoch.
    /// Note: this can be expensive, try to reuse it.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// let guard = tree.pin();
    /// ```
    #[inline]
    pub fn pin(&self) -> epoch::Guard {
        crossbeam_epoch::pin()
    }

    /// Create an empty [ArtUsize] tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        ArtUsize {
            inner: RawTree::new(),
        }
    }

    /// Removes key-value pair from the tree, returns the value if the key was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// let removed = tree.remove(&1, &guard);
    /// assert_eq!(removed, Some(42));
    /// assert!(tree.get(&1, &guard).is_none());
    /// ```
    #[inline]
    pub fn remove(&self, k: &usize, guard: &epoch::Guard) -> Option<usize> {
        let key = UsizeKey::key_from(*k);
        self.inner.remove(&key, guard)
    }

    /// Insert a key-value pair to the tree, returns the previous value if the key was already present.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), 42);
    /// let old = tree.insert(1, 43, &guard);
    /// assert_eq!(old, Some(42));
    /// ```
    #[inline]
    pub fn insert(&self, k: usize, v: usize, guard: &epoch::Guard) -> Option<usize> {
        let key = UsizeKey::key_from(k);
        self.inner.insert(key, v, guard)
    }

    /// Scan the tree with the range of [start, end], write the result to the
    /// `result` buffer.
    /// It scans the length of `result` or the number of the keys within the range, whichever is smaller;
    /// returns the number of the keys scanned.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::ArtUsize;
    /// let tree = ArtUsize::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    ///
    /// let low_key = 1;
    /// let high_key = 2;
    /// let mut result = [(0, 0); 2];
    /// let scanned = tree.range(&low_key, &high_key, &mut result, &guard);
    /// assert_eq!(scanned, 1);
    /// assert_eq!(result, [(1, 42), (0, 0)]);
    /// ```
    #[inline]
    pub fn range(
        &self,
        start: &usize,
        end: &usize,
        result: &mut [(usize, usize)],
        guard: &epoch::Guard,
    ) -> usize {
        let start = UsizeKey::key_from(*start);
        let end = UsizeKey::key_from(*end);
        self.inner.range(&start, &end, result, guard)
    }

    #[inline]
    pub fn compute_if_present(
        &self,
        key: &usize,
        guard: &epoch::Guard,
        f: impl Fn(usize, usize) -> usize,
    ) -> Option<usize> {
        let u_key = UsizeKey::key_from(*key);

        let remapped = |v: usize| -> usize { f(*key, v) };
        self.inner.compute_if_present(&u_key, remapped, guard)
    }

    /// Display the internal node statistics
    #[cfg(feature = "stats")]
    pub fn stats(&self) -> stats::NodeStats {
        self.inner.stats()
    }
}

/// The main adaptive radix tree.
pub struct Art<V: Clone> {
    inner: ManuallyDrop<RawTree<UsizeKey>>,
    pt: std::marker::PhantomData<V>,
}

impl<V: Clone> Default for Art<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> Drop for Art<V> {
    fn drop(&mut self) {
        let v = unsafe { std::mem::ManuallyDrop::take(&mut self.inner.root) };

        let mut sub_nodes = vec![(Box::into_raw(v) as *const BaseNode, 0)];

        while !sub_nodes.is_empty() {
            let (node, level) = sub_nodes.pop().unwrap();
            let children = unsafe { &*node }.get_children(0, 255);
            for (_k, n) in children {
                if level != 7 {
                    sub_nodes.push((
                        n.as_ptr(),
                        level + 1 + unsafe { &*n.as_ptr() }.prefix().len(),
                    ));
                } else {
                    let payload = n.as_tid() as *mut V;
                    unsafe { std::mem::drop(Box::from_raw(payload)) };
                }
            }
            unsafe {
                std::ptr::drop_in_place(node as *mut BaseNode);
            }
        }
    }
}

impl<V: Clone> Art<V> {
    /// Create an empty [Art] tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::<String>::new();
    /// ```
    pub fn new() -> Self {
        Art {
            inner: ManuallyDrop::new(RawTree::new()),
            pt: std::marker::PhantomData,
        }
    }

    /// Enters an epoch.
    /// Note: this can be expensive, try to reuse it.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::<String>::new();
    /// let guard = tree.pin();
    /// ```
    #[inline]
    pub fn pin(&self) -> epoch::Guard {
        crossbeam_epoch::pin()
    }

    /// Insert a key-value pair to the tree, return the previous value if the key was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, "42".to_string(), &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), "42".to_string());
    /// let old = tree.insert(1, "43".to_string(), &guard);
    /// assert_eq!(old, Some("42".to_string()));
    /// ```
    pub fn insert(&self, k: usize, v: V, guard: &epoch::Guard) -> Option<V> {
        let key = UsizeKey::key_from(k);
        let boxed = Box::new(v);
        let boxed_ptr = Box::into_raw(boxed) as *mut V;
        let old = self.inner.insert(key, boxed_ptr as usize, guard)?;
        let val = unsafe { Box::from_raw(old as *mut V) };
        let val = *val;
        Some(val)
    }

    /// Returns a copy of the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, "Support Ukraine".to_string(), &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), "Support Ukraine".to_string());
    /// ```
    pub fn get(&self, key: &usize, guard: &epoch::Guard) -> Option<V> {
        let key = UsizeKey::key_from(*key);
        let addr = self.inner.get(&key, guard)?;
        let addr = addr as *const V;
        unsafe { Some((&*addr).clone()) }
    }

    /// Removes key-value pair from the tree, returns the value if the key was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, "Hello world!", &guard);
    /// let val = tree.remove(&1, &guard);
    /// assert_eq!(val, Some("Hello world!"));
    /// assert!(tree.get(&1, &guard).is_none());
    /// ```
    pub fn remove(&self, k: &usize, guard: &epoch::Guard) -> Option<V> {
        let key = UsizeKey::key_from(*k);

        let addr = self.inner.remove(&key, guard)?;
        let val = unsafe { (addr as *mut V).read() };

        guard.defer(move || {
            let addr = addr as *mut V;
            unsafe { Box::from_raw(addr) };
        });
        Some(val)
    }

    /// Scan the tree with the range of [start, end], write the result to the
    /// `result` buffer.
    /// It scans the length of `result` or the number of the keys within the range, whichever is smaller;
    /// returns the number of the keys scanned.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::new();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, "Usize", &guard);
    ///
    /// let low_key = 1;
    /// let high_key = 2;
    /// let mut result = [(0, ""); 2];
    /// let scanned = tree.range(&low_key, &high_key, &mut result, &guard);
    /// assert_eq!(scanned, 1);
    /// assert_eq!(result, [(1, "Usize"), (0, "")]);
    /// ```
    pub fn range(
        &self,
        start: &usize,
        end: &usize,
        result: &mut [(usize, V)],
        guard: &epoch::Guard,
    ) -> usize {
        let start = UsizeKey::key_from(*start);
        let end = UsizeKey::key_from(*end);
        let mut result_tmp: Vec<(usize, usize)> = vec![(0, 0); result.len()];
        let result_cnt = self.inner.range(&start, &end, &mut result_tmp, guard);
        for (idx, r) in result_tmp.iter().take(result_cnt).enumerate() {
            let addr = r.1 as *const V;
            unsafe {
                result[idx] = (r.0, (*addr).clone());
            }
        }
        result_cnt
    }
}
