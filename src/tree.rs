use std::marker::PhantomData;

use crossbeam_epoch::Guard;
use douhua::MemType;
use rand::{thread_rng, Rng};

use crate::{
    base_node::{BaseNode, Node, NodeType, Prefix, MAX_KEY_LEN},
    error::{ArtError, OOMError},
    key::RawKey,
    lock::ReadGuard,
    node_16::Node16,
    node_256::Node256,
    node_4::Node4,
    node_48::Node48,
    node_ptr::NodePtr,
    range_scan::RangeScan,
    utils::{AllocPolicy, Backoff, PrefixKeysTracker},
    CongeeAllocator, DefaultAllocator,
};

/// Raw interface to the ART tree.
/// The `Art` is a wrapper around the `RawArt` that provides a safe interface.
/// Unlike `Art`, it support arbitrary `Key` types, see also `RawKey`.
pub(crate) struct RawTree<K: RawKey, A: CongeeAllocator + Clone + 'static = DefaultAllocator> {
    pub(crate) root: *const Node256,
    allocator: A,
    migration_map: dashmap::DashMap<PrefixKeysTracker, *const BaseNode>,
    _pt_key: PhantomData<K>,
}

unsafe impl<K: RawKey, A: CongeeAllocator + Clone> Send for RawTree<K, A> {}
unsafe impl<K: RawKey, A: CongeeAllocator + Clone> Sync for RawTree<K, A> {}

impl<K: RawKey> Default for RawTree<K> {
    fn default() -> Self {
        Self::new(DefaultAllocator {})
    }
}

impl<T: RawKey, A: CongeeAllocator + Clone> Drop for RawTree<T, A> {
    fn drop(&mut self) {
        let mut sub_nodes = vec![(self.root as *const BaseNode, 0)];

        while !sub_nodes.is_empty() {
            let (node, level) = sub_nodes.pop().unwrap();

            let children = unsafe { &*node }.get_children(0, 255);
            for (_k, n) in children {
                if level != (MAX_KEY_LEN - 1) {
                    sub_nodes.push((n.as_ptr(), unsafe { &*n.as_ptr() }.prefix().len()));
                }
            }
            unsafe {
                BaseNode::drop_node(node as *mut BaseNode, self.allocator.clone());
            }
        }
    }
}

impl<T: RawKey, A: CongeeAllocator + Clone> RawTree<T, A> {
    pub fn new(allocator: A) -> Self {
        RawTree {
            root: BaseNode::make_node::<Node256>(&[], &allocator, AllocPolicy::LocalOnly)
                .expect("Can't allocate memory for root node!") as *const Node256,
            allocator,
            migration_map: dashmap::DashMap::new(),
            _pt_key: PhantomData,
        }
    }
}

impl<T: RawKey, A: CongeeAllocator + Clone> RawTree<T, A> {
    #[inline]
    pub(crate) fn get(&self, key: &T, guard: &Guard) -> Option<usize> {
        let backoff = Backoff::new();
        loop {
            match self.compute_if_present_inner(key, &mut Some, guard) {
                Ok(n) => {
                    let v = n?;
                    return Some(v.0);
                }
                Err(_) => backoff.spin(),
            }
        }
    }

    #[inline]
    fn insert_inner<F>(
        &self,
        k: &T,
        tid_func: &mut F,
        guard: &Guard,
    ) -> Result<Option<usize>, ArtError>
    where
        F: FnMut(Option<usize>) -> usize,
    {
        let mut parent_node = None;
        let mut next_node = self.root as *const BaseNode;
        let mut parent_key: u8;
        let mut node_key: u8 = 0;
        let mut level = 0;

        let mut node;

        loop {
            parent_key = node_key;

            node = match self.read_check_replace_node(next_node, k, level) {
                Ok(n) => n,
                Err(ArtError::NodeMoved) => {
                    // node moved, we need to check migration map to update the node to new location.
                    todo!("prefix keys not match, need to implement replacement");
                }
                Err(e) => return Err(e),
            };

            let mut next_level = level;
            let res = self.check_prefix_not_match(node.as_ref(), k, &mut next_level);
            match res {
                None => {
                    level = next_level;
                    node_key = k.as_bytes()[level];

                    let next_node_tmp = node.as_ref().get_child(node_key);

                    node.check_version()?;

                    let next_node_tmp = if let Some(n) = next_node_tmp {
                        n
                    } else {
                        let new_leaf = {
                            if level == (MAX_KEY_LEN - 1) {
                                // last key, just insert the tid
                                NodePtr::from_tid(tid_func(None))
                            } else {
                                let new_prefix = k.as_bytes();
                                let n4 = BaseNode::make_node::<Node4>(
                                    &new_prefix[..k.len() - 1],
                                    &self.allocator,
                                    AllocPolicy::BestEffort,
                                )?;
                                unsafe { &mut *n4 }.insert(
                                    k.as_bytes()[k.len() - 1],
                                    NodePtr::from_tid(tid_func(None)),
                                );
                                NodePtr::from_node(n4 as *mut BaseNode)
                            }
                        };

                        if let Err(e) = BaseNode::insert_and_unlock(
                            node,
                            (parent_key, parent_node),
                            (node_key, new_leaf),
                            &self.allocator,
                            guard,
                        ) {
                            if level != (MAX_KEY_LEN - 1) {
                                unsafe {
                                    BaseNode::drop_node(
                                        new_leaf.as_ptr() as *mut BaseNode,
                                        self.allocator.clone(),
                                    );
                                }
                            }
                            return Err(e);
                        }

                        return Ok(None);
                    };

                    if let Some(p) = parent_node {
                        p.unlock()?;
                    }

                    if level == (MAX_KEY_LEN - 1) {
                        // At this point, the level must point to the last u8 of the key,
                        // meaning that we are updating an existing value.

                        let old = node.as_ref().get_child(node_key).unwrap().as_tid();
                        let new = tid_func(Some(old));
                        if old == new {
                            node.check_version()?;
                            return Ok(Some(old));
                        }

                        let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                        let old = write_n.as_mut().change(node_key, NodePtr::from_tid(new));
                        return Ok(Some(old.as_tid()));
                    }
                    next_node = next_node_tmp.as_ptr();
                    level += 1;
                }

                Some(no_match_key) => {
                    let mut write_p = parent_node.unwrap().upgrade().map_err(|(_n, v)| v)?;
                    let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                    // 1) Create new node which will be parent of node, Set common prefix, level to this node
                    // let prefix_len = write_n.as_ref().prefix().len();
                    let new_middle_node = BaseNode::make_node::<Node4>(
                        write_n.as_ref().prefix()[0..next_level].as_ref(),
                        &self.allocator,
                        AllocPolicy::BestEffort,
                    )?;

                    // 2)  add node and (tid, *k) as children
                    if next_level == (MAX_KEY_LEN - 1) {
                        // this is the last key, just insert to node
                        unsafe { &mut *new_middle_node }
                            .insert(k.as_bytes()[next_level], NodePtr::from_tid(tid_func(None)));
                    } else {
                        // otherwise create a new node
                        let single_new_node = BaseNode::make_node::<Node4>(
                            &k.as_bytes()[..k.len() - 1],
                            &self.allocator,
                            AllocPolicy::BestEffort,
                        )?;

                        unsafe { &mut *single_new_node }
                            .insert(k.as_bytes()[k.len() - 1], NodePtr::from_tid(tid_func(None)));
                        unsafe { &mut *new_middle_node }.insert(
                            k.as_bytes()[next_level],
                            NodePtr::from_node(single_new_node as *const BaseNode),
                        );
                    }

                    unsafe { &mut *new_middle_node }
                        .insert(no_match_key, NodePtr::from_node(write_n.as_mut()));

                    // 3) update parentNode to point to the new node, unlock
                    write_p.as_mut().change(
                        parent_key,
                        NodePtr::from_node(new_middle_node as *mut BaseNode),
                    );

                    return Ok(None);
                }
            }
            parent_node = Some(node);
        }
    }

    #[inline]
    pub(crate) fn insert(
        &self,
        k: T,
        tid: usize,
        guard: &Guard,
    ) -> Result<Option<usize>, OOMError> {
        let backoff = Backoff::new();
        loop {
            match self.insert_inner(&k, &mut |_| tid, guard) {
                Ok(v) => return Ok(v),
                Err(e) => match e {
                    ArtError::Locked(_) | ArtError::VersionNotMatch(_) | ArtError::NeedRestart => {
                        backoff.spin();
                        continue;
                    }
                    ArtError::Oom => return Err(OOMError::new()),
                    ArtError::NodeMoved => unreachable!(),
                },
            }
        }
    }

    #[inline]
    pub(crate) fn compute_or_insert<F>(
        &self,
        k: T,
        insert_func: &mut F,
        guard: &Guard,
    ) -> Result<Option<usize>, OOMError>
    where
        F: FnMut(Option<usize>) -> usize,
    {
        let backoff = Backoff::new();
        loop {
            match self.insert_inner(&k, insert_func, guard) {
                Ok(v) => return Ok(v),
                Err(e) => match e {
                    ArtError::Locked(_) | ArtError::VersionNotMatch(_) | ArtError::NeedRestart => {
                        backoff.spin();
                        continue;
                    }
                    ArtError::Oom => return Err(OOMError::new()),
                    ArtError::NodeMoved => unreachable!(),
                },
            }
        }
    }

    #[inline]
    fn check_prefix(node: &BaseNode, key: &T, mut level: usize) -> Option<usize> {
        let n_prefix = node.prefix();
        let k_prefix = key.as_bytes();
        let k_iter = k_prefix.iter().skip(level);

        for (n, k) in n_prefix.iter().skip(level).zip(k_iter) {
            if n != k {
                return None;
            }
            level += 1;
        }
        Some(level)
    }

    #[inline]
    fn check_prefix_not_match(&self, n: &BaseNode, key: &T, level: &mut usize) -> Option<u8> {
        let n_prefix = n.prefix();
        if !n_prefix.is_empty() {
            let p_iter = n_prefix.iter().skip(*level);
            for (i, v) in p_iter.enumerate() {
                if *v != key.as_bytes()[*level] {
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
        start: &T,
        end: &T,
        result: &mut [(usize, usize)],
        _guard: &Guard,
    ) -> usize {
        let mut range_scan = RangeScan::new(start, end, result, self.root as *const BaseNode);

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

    fn promote_node<'a>(&self, node: ReadGuard) -> Result<ReadGuard<'a>, ArtError> {
        let mut write_n = node.upgrade().map_err(|(_, v)| v)?;

        let new_node = match write_n.as_ref().meta.node_type {
            NodeType::N4 => {
                let node = BaseNode::make_node::<Node4>(
                    write_n.as_ref().prefix(),
                    &self.allocator,
                    AllocPolicy::LocalOnly,
                )?;
                unsafe {
                    { &*(write_n.as_ref() as *const BaseNode as *const Node4) }.copy_to(&mut *node);
                }
                unsafe { (&*node).base().read_lock().unwrap() }
            }
            NodeType::N16 => {
                let node = BaseNode::make_node::<Node16>(
                    write_n.as_ref().prefix(),
                    &self.allocator,
                    AllocPolicy::LocalOnly,
                )?;
                unsafe {
                    { &*(write_n.as_ref() as *const BaseNode as *const Node16) }
                        .copy_to(&mut *node);
                }
                unsafe { (&*node).base().read_lock().unwrap() }
            }
            NodeType::N48 => {
                let node = BaseNode::make_node::<Node48>(
                    write_n.as_ref().prefix(),
                    &self.allocator,
                    AllocPolicy::LocalOnly,
                )?;
                unsafe {
                    { &*(write_n.as_ref() as *const BaseNode as *const Node48) }
                        .copy_to(&mut *node);
                }
                unsafe { (&*node).base().read_lock().unwrap() }
            }
            NodeType::N256 => {
                let node = BaseNode::make_node::<Node256>(
                    write_n.as_ref().prefix(),
                    &self.allocator,
                    AllocPolicy::LocalOnly,
                )?;
                unsafe {
                    { &*(write_n.as_ref() as *const BaseNode as *const Node256) }
                        .copy_to(&mut *node);
                }
                unsafe { (&*node).base().read_lock().unwrap() }
            }
        };
        let write_n_prefix = write_n
            .as_ref()
            .prefix_keys(write_n.as_ref().meta.prefix_cnt as usize);
        write_n.as_mut().meta.prefix_cnt = 0; // has to invalidate the node
        write_n.mark_obsolete();
        self.migration_map
            .insert(write_n_prefix, new_node.as_ref() as *const BaseNode);

        Ok(new_node)
    }

    #[inline]
    fn read_check_replace_node<'a>(
        &self,
        next_ptr: *const BaseNode,
        key: &T,
        level: usize,
    ) -> Result<ReadGuard<'a>, ArtError> {
        let expected_node_id = PrefixKeysTracker::from_raw_key(key, level);
        let actual_node_id = unsafe { &*next_ptr }.prefix_keys(level);
        if expected_node_id != actual_node_id {
            return Err(ArtError::NodeMoved);
        }

        let mut node = unsafe { &*next_ptr }.read_lock()?;

        // Need to check twice here, because the node may be changed after we get the lock
        // The read version will handle future checks.
        if node.as_ref().prefix_keys(level) != expected_node_id {
            return Err(ArtError::NodeMoved);
        }

        // Condition for promoting a node to local memory.
        if node.as_ref().meta.mem_type == MemType::NUMA || thread_rng().gen_range(0..100) < 5 {
            node = match self.promote_node(node) {
                Ok(v) => v,
                Err(ArtError::Oom) => {
                    todo!("Need to evict Local nodes to make room for promotion")
                }
                Err(e) => return Err(e),
            }
        }

        node.as_ref().mark_referenced();

        Ok(node)
    }

    #[inline]
    fn compute_if_present_inner<'a, F>(
        &'a self,
        k: &'a T,
        remapping_function: &mut F,
        guard: &Guard,
    ) -> Result<Option<(usize, Option<usize>)>, ArtError>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let mut parent: Option<(ReadGuard, u8)> = None;
        let mut node_key: u8;
        let mut level = 0;
        let mut node = unsafe { &*self.root }.base().read_lock()?;

        loop {
            level = if let Some(v) = Self::check_prefix(node.as_ref(), k, level) {
                v
            } else {
                return Ok(None);
            };

            node_key = k.as_bytes()[level];

            let child_node = node.as_ref().get_child(node_key);
            node.check_version()?;

            let child_node = match child_node {
                Some(n) => n,
                None => return Ok(None),
            };

            if level == (MAX_KEY_LEN - 1) {
                let tid = child_node.as_tid();
                let new_v = remapping_function(tid);

                match new_v {
                    Some(new_v) => {
                        if new_v == tid {
                            // the value is not change, early return;
                            return Ok(Some((tid, Some(tid))));
                        }
                        let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;
                        let old = write_n
                            .as_mut()
                            .change(k.as_bytes()[level], NodePtr::from_tid(new_v));

                        debug_assert_eq!(tid, old.as_tid());

                        return Ok(Some((old.as_tid(), Some(new_v))));
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
                                BaseNode::drop_node(write_n.as_mut(), allocator);
                                std::mem::forget(write_n);
                            });
                        } else {
                            let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                            write_n.as_mut().remove(node_key);
                        }
                        return Ok(Some((child_node.as_tid(), None)));
                    }
                }
            }

            level += 1;
            let next_node = match self.read_check_replace_node(child_node.as_ptr(), k, level) {
                Ok(n) => n,
                Err(ArtError::NodeMoved) => {
                    // node moved, we need to check migration map to update the node to new location.
                    let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;
                    let expected_node_id = PrefixKeysTracker::from_raw_key(k, level);
                    let val = self.migration_map.get(&expected_node_id).expect(
                        "node should be in migration map because we find a node moved error",
                    );
                    let new_child_ptr = *val.value();
                    write_n
                        .as_mut()
                        .change(node_key, NodePtr::from_node(new_child_ptr));
                    return Err(ArtError::NeedRestart);
                }
                Err(e) => return Err(e),
            };

            parent = Some((node, node_key));

            node = next_node;
        }
    }

    #[inline]
    pub(crate) fn compute_if_present<F>(
        &self,
        k: &T,
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

    #[inline]
    #[cfg(feature = "db_extension")]
    pub(crate) fn compute_on_random(
        &self,
        rng: &mut impl rand::Rng,
        f: &mut impl FnMut(usize, usize) -> usize,
        guard: &Guard,
    ) -> Option<(usize, usize, usize)> {
        let backoff = Backoff::new();
        loop {
            match self.compute_on_random_inner(rng, f, guard) {
                Ok(n) => return n,
                Err(_) => backoff.spin(),
            }
        }
    }

    #[inline]
    #[cfg(feature = "db_extension")]
    fn compute_on_random_inner(
        &self,
        rng: &mut impl rand::Rng,
        f: &mut impl FnMut(usize, usize) -> usize,
        _guard: &Guard,
    ) -> Result<Option<(usize, usize, usize)>, ArtError> {
        let mut node = unsafe { &*self.root }.base().read_lock()?;

        let mut key_tracker = crate::utils::PrefixKeysTracker::default();

        loop {
            for k in node.as_ref().prefix() {
                key_tracker.push(*k);
            }

            let child_node = node.as_ref().get_random_child(rng);
            node.check_version()?;

            let (k, child_node) = match child_node {
                Some(n) => n,
                None => return Ok(None),
            };

            key_tracker.push(k);

            if key_tracker.len() == MAX_KEY_LEN {
                let new_v = f(key_tracker.to_usize_key(), child_node.as_tid());
                if new_v == child_node.as_tid() {
                    // Don't acquire the lock if the value is not changed
                    return Ok(Some((key_tracker.to_usize_key(), new_v, new_v)));
                }

                let mut write_n = node.upgrade().map_err(|(_n, v)| v)?;

                let old_v = write_n.as_mut().change(k, NodePtr::from_tid(new_v));

                debug_assert_eq!(old_v.as_tid(), child_node.as_tid());

                return Ok(Some((
                    key_tracker.to_usize_key(),
                    child_node.as_tid(),
                    new_v,
                )));
            }

            node = unsafe { &*child_node.as_ptr() }.read_lock()?;
        }
    }
}
