#![doc = include_str!("../README.md")]
#![allow(clippy::comparison_chain)]
#![allow(clippy::len_without_is_empty)]
#![cfg_attr(doc_cfg, feature(doc_cfg))]
#![feature(slice_ptr_get)]

mod base_node;
mod error;
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

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

use douhua::AllocError;
use douhua::MemType;
use error::OOMError;
use key::RawKey;
use key::UsizeKey;
use tree::RawTree;

/// Types needed to safely access shared data concurrently.
pub mod epoch {
    pub use crossbeam_epoch::{pin, Guard};
}

#[derive(Clone)]
pub struct DefaultAllocator {}

/// # Safety
/// Please check: https://doc.rust-lang.org/std/alloc/trait.Allocator.html
pub unsafe trait CongeeAllocator: Send + Sync {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
        mem_type: MemType,
    ) -> Result<std::ptr::NonNull<[u8]>, AllocError>;

    /// # Safety
    /// Please check: https://doc.rust-lang.org/std/alloc/trait.Allocator.html
    unsafe fn deallocate(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        mem_type: douhua::MemType,
    );

    fn allocate_zeroed(
        &self,
        layout: Layout,
        mem_type: MemType,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let mut ptr = self.allocate(layout, mem_type)?;
        // SAFETY: `alloc` returns a valid memory block
        unsafe { ptr.as_mut().as_mut_ptr().write_bytes(0, ptr.len()) }
        Ok(ptr)
    }
}

unsafe impl CongeeAllocator for DefaultAllocator {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
        _mem_type: MemType,
    ) -> Result<std::ptr::NonNull<[u8]>, AllocError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr_slice = std::ptr::slice_from_raw_parts_mut(ptr, layout.size());
        Ok(std::ptr::NonNull::new(ptr_slice).unwrap())
    }

    unsafe fn deallocate(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        _mem_type: douhua::MemType,
    ) {
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}

/// The adaptive radix tree.
/// Currently we only support only one type of memory, the allocator must return the type of memory requested.
pub struct Art<
    K: Clone + From<usize>,
    V: Clone + From<usize>,
    A: CongeeAllocator + Clone + 'static = DefaultAllocator,
> where
    usize: From<K>,
    usize: From<V>,
{
    inner: RawTree<UsizeKey, A>,
    pt_key: PhantomData<K>,
    pt_val: PhantomData<V>,
}

impl<K: Clone + From<usize>, V: Clone + From<usize>> Default for Art<K, V>
where
    usize: From<K>,
    usize: From<V>,
{
    fn default() -> Self {
        Self::new(DefaultAllocator {})
    }
}

impl<K: Clone + From<usize>, V: Clone + From<usize>, A: CongeeAllocator + Clone> Art<K, V, A>
where
    usize: From<K>,
    usize: From<V>,
{
    /// Returns a copy of the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), 42);
    /// ```
    #[inline]
    pub fn get(&self, key: &K, guard: &epoch::Guard) -> Option<V> {
        let key = UsizeKey::key_from(usize::from(key.clone()));
        let v = self.inner.get(&key, guard)?;
        Some(V::from(v))
    }

    /// Enters an epoch.
    /// Note: this can be expensive, try to reuse it.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::<usize, usize>::default();
    /// let guard = tree.pin();
    /// ```
    #[inline]
    pub fn pin(&self) -> epoch::Guard {
        crossbeam_epoch::pin()
    }

    /// Create an empty [Art] tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::<usize, usize>::default();
    /// ```
    #[inline]
    pub fn new(allocator: A) -> Self {
        Art {
            inner: RawTree::new(allocator),
            pt_key: PhantomData,
            pt_val: PhantomData,
        }
    }

    /// Removes key-value pair from the tree, returns the value if the key was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// let removed = tree.remove(&1, &guard);
    /// assert_eq!(removed, Some(42));
    /// assert!(tree.get(&1, &guard).is_none());
    /// ```
    #[inline]
    pub fn remove(&self, k: &K, guard: &epoch::Guard) -> Option<V> {
        let key = UsizeKey::key_from(usize::from(k.clone()));
        let (old, new) = self.inner.compute_if_present(&key, &mut |_v| None, guard)?;
        debug_assert!(new.is_none());
        Some(V::from(old))
    }

    /// Insert a key-value pair to the tree, returns the previous value if the key was already present.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// assert_eq!(tree.get(&1, &guard).unwrap(), 42);
    /// let old = tree.insert(1, 43, &guard).unwrap();
    /// assert_eq!(old, Some(42));
    /// ```
    #[inline]
    pub fn insert(&self, k: K, v: V, guard: &epoch::Guard) -> Result<Option<V>, OOMError> {
        let key = UsizeKey::key_from(usize::from(k));
        let val = self.inner.insert(key, usize::from(v), guard);
        val.map(|inner| inner.map(|v| V::from(v)))
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
    /// let tree = Art::default();
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
        start: &K,
        end: &K,
        result: &mut [(usize, usize)],
        guard: &epoch::Guard,
    ) -> usize {
        let start = UsizeKey::key_from(usize::from(start.clone()));
        let end = UsizeKey::key_from(usize::from(end.clone()));
        self.inner.range(&start, &end, result, guard)
    }

    /// Compute and update the value if the key presents in the tree.
    /// Returns the (old, new) value
    ///
    /// Note that the function `f` is a FnMut and it must be safe to execute multiple times.
    /// The `f` is expected to be short and fast as it will hold a exclusive lock on the leaf node.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// let old = tree.compute_if_present(&1, |v| Some(v+1), &guard).unwrap();
    /// assert_eq!(old, (42, Some(43)));
    /// let val = tree.get(&1, &guard).unwrap();
    /// assert_eq!(val, 43);
    /// ```
    #[inline]
    pub fn compute_if_present<F>(
        &self,
        key: &K,
        mut f: F,
        guard: &epoch::Guard,
    ) -> Option<(usize, Option<usize>)>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let u_key = UsizeKey::key_from(usize::from(key.clone()));

        self.inner.compute_if_present(&u_key, &mut f, guard)
    }

    /// Compute or insert the value if the key is not in the tree.
    /// Returns the Option(old) value
    ///
    /// Note that the function `f` is a FnMut and it must be safe to execute multiple times.
    /// The `f` is expected to be short and fast as it will hold a exclusive lock on the leaf node.
    ///
    /// # Examples
    ///
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    ///
    /// tree.insert(1, 42, &guard);
    /// let old = tree.compute_or_insert(1, |v| v.unwrap() + 1, &guard).unwrap().unwrap();
    /// assert_eq!(old, 42);
    /// let val = tree.get(&1, &guard).unwrap();
    /// assert_eq!(val, 43);
    ///
    /// let old = tree.compute_or_insert(2, |v| {
    ///     assert!(v.is_none());
    ///     2
    /// }, &guard).unwrap();
    /// assert!(old.is_none());
    /// let val = tree.get(&2, &guard).unwrap();
    /// assert_eq!(val, 2);
    /// ```
    pub fn compute_or_insert<F>(
        &self,
        key: K,
        mut f: F,
        guard: &epoch::Guard,
    ) -> Result<Option<V>, OOMError>
    where
        F: FnMut(Option<usize>) -> usize,
    {
        let u_key = UsizeKey::key_from(usize::from(key));
        let u_val = self.inner.compute_or_insert(u_key, &mut f, guard)?;
        Ok(u_val.map(|v| V::from(v)))
    }

    /// Display the internal node statistics
    #[cfg(feature = "stats")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "stats")))]
    pub fn stats(&self) -> stats::NodeStats {
        self.inner.stats()
    }

    /// Get a random value from the tree, perform the transformation `f`.
    /// This is useful for randomized algorithms.
    ///
    /// `f` takes key and value as input and return the new value, |key: usize, value: usize| -> usize.
    ///
    /// Returns (key, old_value, new_value)
    ///
    /// Note that the function `f` is a FnMut and it must be safe to execute multiple times.
    /// The `f` is expected to be short and fast as it will hold a exclusive lock on the leaf node.
    ///
    /// # Examples:
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    /// tree.insert(1, 42, &guard);
    /// let mut rng = rand::thread_rng();
    /// let (key, old_v, new_v) = tree.compute_on_random(&mut rng, |k, v| {
    ///     assert_eq!(k, 1);
    ///     assert_eq!(v, 42);
    ///     v + 1
    /// }, &guard).unwrap();
    /// assert_eq!(key, 1);
    /// assert_eq!(old_v, 42);
    /// assert_eq!(new_v, 43);
    /// ```
    #[cfg(feature = "db_extension")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "db_extension")))]
    pub fn compute_on_random(
        &self,
        rng: &mut impl rand::Rng,
        mut f: impl FnMut(K, V) -> V,
        guard: &epoch::Guard,
    ) -> Option<(K, V, V)> {
        let mut remapped = |key: usize, value: usize| -> usize {
            let v = f(K::from(key), V::from(value));
            usize::from(v)
        };
        let (key, old_v, new_v) = self.inner.compute_on_random(rng, &mut remapped, guard)?;
        Some((K::from(key), V::from(old_v), V::from(new_v)))
    }

    /// Update the value if the old value matches with the new one.
    /// Returns the current value.
    ///
    /// # Examples:
    /// ```
    /// use congee::Art;
    /// let tree = Art::default();
    /// let guard = tree.pin();
    /// tree.insert(1, 42, &guard);
    ///
    ///
    /// let v = tree.compare_exchange(&1, &42, Some(43), &guard).unwrap();
    /// assert_eq!(v, Some(43));
    /// ```
    pub fn compare_exchange(
        &self,
        key: &K,
        old: &V,
        new: Option<V>,
        guard: &epoch::Guard,
    ) -> Result<Option<V>, Option<V>> {
        let u_key = UsizeKey::key_from(usize::from(key.clone()));
        let new_v = new.clone().map(|v| usize::from(v));
        let mut fc = |v: usize| -> Option<usize> {
            if v == usize::from(old.clone()) {
                new_v
            } else {
                Some(v)
            }
        };
        let v = self.inner.compute_if_present(&u_key, &mut fc, guard);
        match v {
            Some((actual_old, actual_new)) => {
                if actual_old == usize::from(old.clone()) && actual_new == new_v {
                    Ok(new)
                } else {
                    Err(actual_new.map(|v| V::from(v)))
                }
            }
            None => Err(None),
        }
    }
}
