use crate::{base_node::BaseNode, child_ptr::ChildPtr, key::Key, lock::ReadGuard};

enum PrefixCheckEqualsResult {
    BothMatch,
    Contained,
    NotMatch,
}

enum PrefixCompareResult {
    Smaller,
    Equal,
    Bigger,
}

pub(crate) struct RangeScan<'a, T: Key> {
    start: &'a T,
    end: &'a T,
    result: &'a mut [usize],
    root: *const BaseNode,
    to_continue: usize,
    result_found: usize,
}

#[derive(Default, Clone)]
pub(crate) struct KeyTracker {
    len: usize,
    data: [u8; 8],
}

impl KeyTracker {
    #[inline]
    pub(crate) fn push(&mut self, key: u8) {
        debug_assert!(self.len <= 8);

        self.data[self.len as usize] = key;
        self.len += 1;
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> u8 {
        debug_assert!(self.len > 0);

        let v = self.data[self.len as usize - 1];
        self.len -= 1;
        v
    }

    pub(crate) fn to_usize_key(&self) -> usize {
        assert!(self.len == 8);
        let val = unsafe { *((&self.data) as *const [u8; 8] as *const usize) };
        std::intrinsics::bswap(val)
    }

    pub(crate) fn append_prefix(node: ChildPtr, key_tracker: &KeyTracker) -> KeyTracker {
        let mut cur_key = key_tracker.clone();
        if node.is_leaf() {
            cur_key
        } else {
            let node_ref = unsafe { &*node.to_ptr() };
            for i in 0..node_ref.get_prefix_len() {
                cur_key.push(node_ref.get_prefix()[i as usize]);
            }
            cur_key
        }
    }
}

impl<'a, T: Key> RangeScan<'a, T> {
    pub(crate) fn new(
        start: &'a T,
        end: &'a T,
        result: &'a mut [usize],
        root: *const BaseNode,
    ) -> Self {
        Self {
            start,
            end,
            result,
            root,
            to_continue: 0,
            result_found: 0,
        }
    }

    fn is_valid_key_pair(&self) -> bool {
        self.start < self.end
    }

    fn key_in_range(&self, key: &KeyTracker) -> bool {
        debug_assert_eq!(key.len, 8);
        let cur_key = key.to_usize_key();

        let start_key =
            std::intrinsics::bswap(unsafe { *(self.start.as_bytes().as_ptr() as *const usize) });
        let end_key =
            std::intrinsics::bswap(unsafe { *(self.end.as_bytes().as_ptr() as *const usize) });

        if start_key <= cur_key && cur_key < end_key {
            return true;
        }
        false
    }

    pub(crate) fn scan(&mut self) -> Option<usize> {
        if !self.is_valid_key_pair() {
            return None;
        }

        'outer: loop {
            let mut level = 0;
            let mut node: ReadGuard;
            let mut next_node = self.root;
            let mut parent_node: Option<ReadGuard> = None;
            self.to_continue = 0;
            self.result_found = 0;

            let mut key_tracker = KeyTracker::default();

            'inner: loop {
                node = if let Ok(v) = unsafe { &*next_node }.read_lock() {
                    v
                } else {
                    continue 'outer;
                };

                let prefix_check_result =
                    match self.check_prefix_equals(node.as_ref(), &mut level, &mut key_tracker) {
                        Ok(v) => v,
                        Err(_) => continue 'outer,
                    };

                if parent_node.is_some() && parent_node.as_ref().unwrap().check_version().is_err() {
                    continue 'outer;
                }

                if node.check_version().is_err() {
                    continue 'outer;
                }

                match prefix_check_result {
                    PrefixCheckEqualsResult::BothMatch => {
                        let start_level = if self.start.len() > level as usize {
                            self.start.as_bytes()[level as usize]
                        } else {
                            0
                        };
                        let end_level = if self.end.len() > level as usize {
                            self.end.as_bytes()[level as usize]
                        } else {
                            255
                        };

                        if start_level != end_level {
                            let children = if let Ok(val) =
                                BaseNode::get_children(&node, start_level, end_level)
                            {
                                val
                            } else {
                                continue 'outer;
                            };

                            for (k, n) in children.iter() {
                                key_tracker.push(*k);
                                if *k == start_level {
                                    if self
                                        .find_start(*n, level + 1, &node, key_tracker.clone())
                                        .is_err()
                                    {
                                        continue 'outer;
                                    };
                                } else if *k > start_level && *k < end_level {
                                    let cur_key = KeyTracker::append_prefix(*n, &key_tracker);
                                    if self.copy_node(*n, &cur_key).is_err() {
                                        continue 'outer;
                                    };
                                } else if *k == end_level {
                                    if self
                                        .find_end(*n, level + 1, &node, key_tracker.clone())
                                        .is_err()
                                    {
                                        continue 'outer;
                                    }
                                }
                                key_tracker.pop();

                                if self.to_continue > 0 {
                                    break 'inner;
                                }
                            }
                        } else {
                            let next_node_tmp = BaseNode::get_child(start_level, node.as_ref())?;
                            if node.check_version().is_err() {
                                continue 'outer;
                            };

                            key_tracker.push(start_level);
                            if next_node_tmp.is_leaf() {
                                if self.copy_node(next_node_tmp, &key_tracker).is_err() {
                                    continue 'outer;
                                };
                                break;
                            }
                            next_node = next_node_tmp.to_ptr();

                            level += 1;
                            parent_node = Some(node);
                            continue;
                        }
                        break;
                    }
                    PrefixCheckEqualsResult::Contained => {
                        if self
                            .copy_node(ChildPtr::from_ptr(node.as_ref()), &key_tracker)
                            .is_err()
                        {
                            continue 'outer;
                        }
                    }
                    PrefixCheckEqualsResult::NotMatch => {
                        return None;
                    }
                }
                break;
            }

            if self.result_found > 0 {
                return Some(self.result_found);
            } else {
                return None;
            }
        }
    }

    fn find_end(
        &mut self,
        node: ChildPtr,
        mut level: u32,
        parent_node: &ReadGuard,
        mut key_tracker: KeyTracker,
    ) -> Result<(), ()> {
        if node.is_leaf() {
            return self.copy_node(node, &key_tracker);
        }

        let node = unsafe { &*node.to_ptr() }.read_lock().map_err(|_| {})?;
        let prefix_result =
            self.check_prefix_compare(node.as_ref(), self.end, 255, &mut level, &mut key_tracker)?;

        parent_node.check_version().map_err(|_| {})?;
        node.check_version().map_err(|_| {})?;

        match prefix_result {
            PrefixCompareResult::Bigger => Ok(()),
            PrefixCompareResult::Equal => {
                let end_level = if self.end.len() > level as usize {
                    self.end.as_bytes()[level as usize]
                } else {
                    255
                };

                let children = BaseNode::get_children(&node, 0, end_level).map_err(|_e| ())?;
                for (k, n) in children.iter() {
                    key_tracker.push(*k);
                    if *k == end_level {
                        self.find_end(*n, level + 1, &node, key_tracker.clone())?;
                    } else if *k < end_level {
                        let cur_key = KeyTracker::append_prefix(*n, &key_tracker);
                        self.copy_node(*n, &cur_key)?;
                    }
                    key_tracker.pop();
                    if self.to_continue != 0 {
                        break;
                    }
                }
                Ok(())
            }
            PrefixCompareResult::Smaller => {
                self.copy_node(ChildPtr::from_ptr(node.as_ref()), &key_tracker)
            }
        }
    }

    fn find_start(
        &mut self,
        node: ChildPtr,
        mut level: u32,
        parent_node: &ReadGuard,
        mut key_tracker: KeyTracker,
    ) -> Result<(), ()> {
        if node.is_leaf() {
            return self.copy_node(node, &key_tracker);
        }

        let node = unsafe { &*node.to_ptr() }.read_lock().map_err(|_| {})?;
        let prefix_result =
            self.check_prefix_compare(node.as_ref(), self.start, 0, &mut level, &mut key_tracker)?;

        parent_node.check_version().map_err(|_| {})?;
        node.check_version().map_err(|_| {})?;

        match prefix_result {
            PrefixCompareResult::Bigger => {
                self.copy_node(ChildPtr::from_ptr(node.as_ref()), &key_tracker)
            }
            PrefixCompareResult::Equal => {
                let start_level = if self.start.len() > level as usize {
                    self.start.as_bytes()[level as usize]
                } else {
                    0
                };
                let children = BaseNode::get_children(&node, start_level, 255).map_err(|_| ())?;

                for (k, n) in children.iter() {
                    key_tracker.push(*k);
                    if *k == start_level {
                        self.find_start(*n, level + 1, &node, key_tracker.clone())?;
                    } else if *k > start_level {
                        let cur_key = KeyTracker::append_prefix(*n, &key_tracker);
                        self.copy_node(*n, &cur_key)?;
                    }
                    key_tracker.pop();
                    if self.to_continue != 0 {
                        break;
                    }
                }
                Ok(())
            }
            PrefixCompareResult::Smaller => Ok(()),
        }
    }

    // FIXME: copy node should check parent version to make sure the node is not changed
    fn copy_node(&mut self, node: ChildPtr, key_tracker: &KeyTracker) -> Result<(), ()> {
        if node.is_leaf() {
            if self.key_in_range(key_tracker) {
                if self.result_found == self.result.len() {
                    self.to_continue = node.to_tid();
                    return Ok(());
                }
                self.result[self.result_found] = node.to_tid();
                self.result_found += 1;
            };
        } else {
            let node = unsafe { &*node.to_ptr() }.read_lock().map_err(|_| ())?;
            let mut key_tracker = key_tracker.clone();

            let children = BaseNode::get_children(&node, 0, 255).map_err(|_| ())?;

            for (k, c) in children.iter() {
                key_tracker.push(*k);

                let cur_key = KeyTracker::append_prefix(*c, &key_tracker);
                self.copy_node(*c, &cur_key)?;

                if self.to_continue != 0 {
                    break;
                }

                key_tracker.pop();
            }
        }
        Ok(())
    }

    fn check_prefix_compare(
        &self,
        n: &BaseNode,
        k: &T,
        fill_key: u8,
        level: &mut u32,
        key_tracker: &mut KeyTracker,
    ) -> Result<PrefixCompareResult, ()> {
        if n.has_prefix() {
            for i in 0..n.get_prefix_len() as usize {
                let k_level = if k.len() as u32 > *level {
                    k.as_bytes()[*level as usize]
                } else {
                    fill_key
                };

                let cur_key = n.get_prefix()[i];
                key_tracker.push(cur_key);

                if cur_key < k_level {
                    for j in (i + 1)..n.get_prefix_len() as usize {
                        key_tracker.push(n.get_prefix()[j]);
                    }
                    return Ok(PrefixCompareResult::Smaller);
                } else if cur_key > k_level {
                    for j in (i + 1)..n.get_prefix_len() as usize {
                        key_tracker.push(n.get_prefix()[j]);
                    }
                    return Ok(PrefixCompareResult::Bigger);
                }

                *level += 1;
            }
        }
        Ok(PrefixCompareResult::Equal)
    }

    fn check_prefix_equals(
        &self,
        n: &BaseNode,
        level: &mut u32,
        key_tracker: &mut KeyTracker,
    ) -> Result<PrefixCheckEqualsResult, ()> {
        if n.has_prefix() {
            for i in 0..n.get_prefix_len() as usize {
                let start_level = if self.start.len() as u32 > *level {
                    self.start.as_bytes()[*level as usize]
                } else {
                    0
                };

                let end_level = if self.end.len() as u32 > *level {
                    self.end.as_bytes()[*level as usize]
                } else {
                    255
                };

                let cur_key = n.get_prefix()[i as usize];

                if (cur_key == start_level) && (cur_key == end_level) {
                    *level += 1;
                    key_tracker.push(cur_key);
                    continue;
                } else if (cur_key >= start_level) && (cur_key <= end_level) {
                    key_tracker.push(cur_key);
                    for j in (i + 1)..n.get_prefix_len() as usize {
                        key_tracker.push(n.get_prefix()[j]);
                    }
                    return Ok(PrefixCheckEqualsResult::Contained);
                } else if cur_key < start_level || cur_key > end_level {
                    return Ok(PrefixCheckEqualsResult::NotMatch);
                }
            }
        }
        Ok(PrefixCheckEqualsResult::BothMatch)
    }
}
