use std::{marker::PhantomData, ptr::NonNull};

use crossbeam_epoch::Guard;

use crate::{
    base_node::{BaseNode, Node, Prefix},
    error::{ArtError, OOMError},
    lock::ReadGuard,
    node_256::Node256,
    node_4::Node4,
    node_ptr::{ChildIsPayload, ChildIsSubNode, NodePtr, PtrType},
    range_scan::RangeScan,
    utils::Backoff,
    Allocator, DefaultAllocator,
};

/// Raw interface to the ART tree.
/// The `Art` is a wrapper around the `RawArt` that provides a safe interface.
pub(crate) struct RawCongee<const K_LEN: usize, A: Allocator + Clone + 'static = DefaultAllocator> {
    pub(crate) root: NonNull<Node256>,
    allocator: A,
    _pt_key: PhantomData<[u8; K_LEN]>,
}

unsafe impl<const K_LEN: usize, A: Allocator + Clone> Send for RawCongee<K_LEN, A> {}
unsafe impl<const K_LEN: usize, A: Allocator + Clone> Sync for RawCongee<K_LEN, A> {}

impl<const K_LEN: usize> Default for RawCongee<K_LEN> {
    fn default() -> Self {
        Self::new(DefaultAllocator {})
    }
}

impl<const K_LEN: usize, A: Allocator + Clone> Drop for RawCongee<K_LEN, A> {
    fn drop(&mut self) {
        let mut sub_nodes = vec![(NodePtr::from_root(self.root), 0)];

        while let Some((node, level)) = sub_nodes.pop() {
            match node.downcast::<K_LEN>(level) {
                PtrType::Payload(_) => {
                    continue;
                }
                PtrType::SubNode(sub_node) => {
                    let node_lock = BaseNode::read_lock(sub_node).unwrap();
                    let children = node_lock.as_ref().get_children(0, 255);
                    for (_k, n) in children {
                        match n.downcast::<K_LEN>(level) {
                            PtrType::Payload(_) => {}
                            PtrType::SubNode(sub_sub_node) => {
                                let node_lock = BaseNode::read_lock(sub_sub_node).unwrap();
                                sub_nodes.push((n, node_lock.as_ref().prefix().len()));
                            }
                        }
                    }
                    unsafe {
                        BaseNode::drop_node(sub_node, self.allocator.clone());
                    }
                }
            }
        }
    }
}

impl<const K_LEN: usize, A: Allocator + Clone> RawCongee<K_LEN, A> {
    pub fn new(allocator: A) -> Self {
        let root = BaseNode::make_node::<Node256>(&[], &allocator)
            .expect("Can't allocate memory for root node!");
        RawCongee {
            root: root.into_non_null(),
            allocator,
            _pt_key: PhantomData,
        }
    }
}

impl<const K_LEN: usize, A: Allocator + Clone + Send> RawCongee<K_LEN, A> {
    #[inline]
    pub(crate) fn get(&self, key: &[u8; K_LEN], _guard: &Guard) -> Option<usize> {
        'outer: loop {
            let mut level = 0;

            let mut node = if let Ok(v) = BaseNode::read_lock_root(self.root) {
                v
            } else {
                continue;
            };

            loop {
                level = Self::check_prefix(node.as_ref(), key, level)?;

                let child_node = node
                    .as_ref()
                    .get_child(unsafe { *key.get_unchecked(level) });
                if node.check_version().is_err() {
                    continue 'outer;
                }

                let child_node = child_node?;

                match child_node.downcast::<K_LEN>(level) {
                    PtrType::Payload(tid) => {
                        return Some(tid);
                    }
                    PtrType::SubNode(sub_node) => {
                        level += 1;

                        node = if let Ok(n) = BaseNode::read_lock(sub_node) {
                            n
                        } else {
                            continue 'outer;
                        };
                    }
                }
            }
        }
    }

    fn is_last_level<'a>(current_level: usize) -> Result<ChildIsPayload<'a>, ChildIsSubNode<'a>> {
        if current_level == (K_LEN - 1) {
            Ok(ChildIsPayload::new())
        } else {
            Err(ChildIsSubNode::new())
        }
    }

    #[inline]
    fn insert_inner<F>(
        &self,
        k: &[u8; K_LEN],
        tid_func: &mut F,
        guard: &Guard,
    ) -> Result<Option<usize>, ArtError>
    where
        F: FnMut(Option<usize>) -> usize,
    {
        let mut parent_node = None;
        let mut node = BaseNode::read_lock_root(self.root)?;
        let mut parent_key: u8;
        let mut node_key: u8 = 0;
        let mut level = 0usize;

        loop {
            parent_key = node_key;

            let mut next_level = level;
            let res = self.check_prefix_not_match(node.as_ref(), k, &mut next_level);
            match res {
                None => {
                    level = next_level;
                    node_key = k[level];

                    let next_node = node.as_ref().get_child(node_key);

                    node.check_version()?;

                    let next_node = if let Some(n) = next_node {
                        n
                    } else {
                        let new_leaf = {
                            match Self::is_last_level(level) {
                                Ok(_is_last_level) => NodePtr::from_payload(tid_func(None)),
                                Err(_is_sub_node) => {
                                    let new_prefix = k;
                                    let mut n4 = BaseNode::make_node::<Node4>(
                                        &new_prefix[..k.len() - 1],
                                        &self.allocator,
                                    )?;
                                    n4.as_mut().insert(
                                        k[k.len() - 1],
                                        NodePtr::from_payload(tid_func(None)),
                                    );
                                    n4.into_note_ptr()
                                }
                            }
                        };

                        if let Err(e) = BaseNode::insert_and_unlock(
                            node,
                            (parent_key, parent_node),
                            (node_key, new_leaf),
                            &self.allocator,
                            guard,
                        ) {
                            match new_leaf.downcast::<K_LEN>(level) {
                                PtrType::Payload(_) => {}
                                PtrType::SubNode(sub_node) => unsafe {
                                    BaseNode::drop_node(sub_node, self.allocator.clone());
                                },
                            }
                            return Err(e);
                        }

                        return Ok(None);
                    };

                    if let Some(p) = parent_node {
                        p.unlock()?;
                    }

                    match next_node.downcast::<K_LEN>(level) {
                        PtrType::Payload(old) => {
                            // At this point, the level must point to the last u8 of the key,
                            // meaning that we are updating an existing value.
                            let new = tid_func(Some(old));
                            if old == new {
                                node.check_version()?;
                                return Ok(Some(old));
                            }

                            let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                            write_n
                                .as_mut()
                                .change(node_key, NodePtr::from_payload(new));
                            return Ok(Some(old));
                        }
                        PtrType::SubNode(sub_node) => {
                            parent_node = Some(node);
                            node = BaseNode::read_lock(sub_node)?;
                            level += 1;
                        }
                    }
                }

                Some(no_match_key) => {
                    let mut write_p = parent_node.unwrap().upgrade().map_err(|(_n, v)| v)?;
                    let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                    // 1) Create new node which will be parent of node, Set common prefix, level to this node
                    // let prefix_len = write_n.as_ref().prefix().len();
                    let mut new_middle_node = BaseNode::make_node::<Node4>(
                        write_n.as_ref().prefix()[0..next_level].as_ref(),
                        &self.allocator,
                    )?;

                    // 2)  add node and (tid, *k) as children
                    if next_level == (K_LEN - 1) {
                        // this is the last key, just insert to node
                        new_middle_node
                            .as_mut()
                            .insert(k[next_level], NodePtr::from_payload(tid_func(None)));
                    } else {
                        // otherwise create a new node
                        let mut single_new_node =
                            BaseNode::make_node::<Node4>(&k[..k.len() - 1], &self.allocator)?;

                        single_new_node
                            .as_mut()
                            .insert(k[k.len() - 1], NodePtr::from_payload(tid_func(None)));
                        new_middle_node
                            .as_mut()
                            .insert(k[next_level], single_new_node.into_note_ptr());
                    }

                    new_middle_node
                        .as_mut()
                        .insert(no_match_key, NodePtr::from_node(write_n.as_mut()));

                    // 3) update parentNode to point to the new node, unlock
                    write_p
                        .as_mut()
                        .change(parent_key, new_middle_node.into_note_ptr());

                    return Ok(None);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn insert(
        &self,
        k: &[u8; K_LEN],
        tid: usize,
        guard: &Guard,
    ) -> Result<Option<usize>, OOMError> {
        let backoff = Backoff::new();
        loop {
            match self.insert_inner(k, &mut |_| tid, guard) {
                Ok(v) => return Ok(v),
                Err(e) => match e {
                    ArtError::Locked | ArtError::VersionNotMatch => {
                        backoff.spin();
                        continue;
                    }
                    ArtError::Oom => return Err(OOMError::new()),
                },
            }
        }
    }

    #[inline]
    pub(crate) fn compute_or_insert<F>(
        &self,
        k: &[u8; K_LEN],
        insert_func: &mut F,
        guard: &Guard,
    ) -> Result<Option<usize>, OOMError>
    where
        F: FnMut(Option<usize>) -> usize,
    {
        let backoff = Backoff::new();
        loop {
            match self.insert_inner(k, insert_func, guard) {
                Ok(v) => return Ok(v),
                Err(e) => match e {
                    ArtError::Locked | ArtError::VersionNotMatch => {
                        backoff.spin();
                        continue;
                    }
                    ArtError::Oom => return Err(OOMError::new()),
                },
            }
        }
    }

    fn check_prefix(node: &BaseNode, key: &[u8; K_LEN], mut level: usize) -> Option<usize> {
        let node_prefix = node.prefix();
        let key_prefix = key;

        for (n, k) in node_prefix.iter().zip(key_prefix).skip(level) {
            if n != k {
                return None;
            }
            level += 1;
        }
        debug_assert!(level == node_prefix.len());
        Some(level)
    }

    #[inline]
    fn check_prefix_not_match(
        &self,
        n: &BaseNode,
        key: &[u8; K_LEN],
        level: &mut usize,
    ) -> Option<u8> {
        let n_prefix = n.prefix();
        if !n_prefix.is_empty() {
            let p_iter = n_prefix.iter().skip(*level);
            for (i, v) in p_iter.enumerate() {
                if *v != key[*level] {
                    let no_matching_key = *v;

                    let mut prefix = Prefix::default();
                    for (j, v) in prefix.iter_mut().enumerate().take(n_prefix.len() - i - 1) {
                        *v = n_prefix[j + 1 + i];
                    }

                    return Some(no_matching_key);
                }
                *level += 1;
            }
        }

        None
    }

    #[inline]
    pub(crate) fn range(
        &self,
        start: &[u8; K_LEN],
        end: &[u8; K_LEN],
        result: &mut [([u8; K_LEN], usize)],
        _guard: &Guard,
    ) -> usize {
        let mut range_scan = RangeScan::new(start, end, result, self.root);

        if !range_scan.is_valid_key_pair() {
            return 0;
        }

        let backoff = Backoff::new();
        loop {
            let scanned = range_scan.scan();
            match scanned {
                Ok(n) => {
                    return n;
                }
                Err(_) => {
                    backoff.spin();
                }
            }
        }
    }

    #[inline]
    fn compute_if_present_inner<F>(
        &self,
        k: &[u8; K_LEN],
        remapping_function: &mut F,
        guard: &Guard,
    ) -> Result<Option<(usize, Option<usize>)>, ArtError>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let mut parent: Option<(ReadGuard, u8)> = None;
        let mut node_key: u8;
        let mut level = 0;
        let mut node = BaseNode::read_lock_root(self.root)?;

        loop {
            level = if let Some(v) = Self::check_prefix(node.as_ref(), k, level) {
                v
            } else {
                return Ok(None);
            };

            node_key = k[level];

            let child_node = node.as_ref().get_child(node_key);
            node.check_version()?;

            let child_node = match child_node {
                Some(n) => n,
                None => return Ok(None),
            };

            match child_node.downcast::<K_LEN>(level) {
                PtrType::Payload(tid) => {
                    let new_v = remapping_function(tid);

                    match new_v {
                        Some(new_v) => {
                            if new_v == tid {
                                // the value is not change, early return;
                                return Ok(Some((tid, Some(tid))));
                            }
                            let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;
                            write_n
                                .as_mut()
                                .change(k[level], NodePtr::from_payload(new_v));

                            return Ok(Some((tid, Some(new_v))));
                        }
                        None => {
                            // new value is none, we need to delete this entry
                            debug_assert!(parent.is_some()); // reaching leaf means we must have parent, bcs root can't be leaf
                            if node.as_ref().get_count() == 1 {
                                let (parent_node, parent_key) = parent.unwrap();
                                let mut write_p = parent_node.upgrade().map_err(|(_n, v)| v)?;

                                let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                                write_p.as_mut().remove(parent_key);

                                write_n.mark_obsolete();
                                let allocator = self.allocator.clone();
                                guard.defer(move || unsafe {
                                    let ptr = NonNull::from(write_n.as_mut());
                                    std::mem::forget(write_n);
                                    BaseNode::drop_node(ptr, allocator);
                                });
                            } else {
                                let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                                write_n.as_mut().remove(node_key);
                            }
                            return Ok(Some((tid, None)));
                        }
                    }
                }
                PtrType::SubNode(sub_node) => {
                    level += 1;
                    parent = Some((node, node_key));
                    node = BaseNode::read_lock(sub_node)?;
                }
            }
        }
    }

    #[inline]
    pub(crate) fn compute_if_present<F>(
        &self,
        k: &[u8; K_LEN],
        remapping_function: &mut F,
        guard: &Guard,
    ) -> Option<(usize, Option<usize>)>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let backoff = Backoff::new();
        loop {
            match self.compute_if_present_inner(k, &mut *remapping_function, guard) {
                Ok(n) => return n,
                Err(_) => backoff.spin(),
            }
        }
    }
}
