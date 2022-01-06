use std::alloc;

use crate::base_node::{BaseNode, Node, NodeType};

#[repr(C)]
pub(crate) struct Node256 {
    base: BaseNode,

    children: [*mut BaseNode; 256],
}

impl Node256 {}

impl Node for Node256 {
    fn new(prefix: *const u8, prefix_len: usize) -> *mut Node256 {
        let layout = alloc::Layout::from_size_align(
            std::mem::size_of::<Node256>(),
            std::mem::align_of::<Node256>(),
        )
        .unwrap();
        let mem = unsafe {
            let mem = alloc::alloc_zeroed(layout) as *mut BaseNode;
            let base = BaseNode::new(NodeType::N256, prefix, prefix_len);
            mem.write(base);
            mem as *mut Node256
        };
        mem
    }

    fn get_children(&self, start: u8, end: u8, children: &mut [*mut BaseNode]) -> (usize, usize) {
        loop {
            let v = if let Ok(v) = self.base.read_lock_or_restart() {
                v
            } else {
                continue;
            };
            let mut child_cnt = 0;

            for i in start..=end {
                if !self.children[i as usize].is_null() {
                    children[child_cnt] = self.children[i as usize];
                    child_cnt += 1;
                }
            }

            if self.base.read_unlock_or_restart(v) {
                continue;
            }

            return (v, child_cnt);
        }
    }

    fn copy_to<N: Node>(&self, dst: *mut N) {
        for i in 0..256 {
            if !self.children[i].is_null() {
                unsafe { &mut *dst }.insert(i as u8, self.children[i]);
            }
        }
    }

    fn base(&self) -> &BaseNode {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BaseNode {
        &mut self.base
    }

    fn is_full(&self) -> bool {
        false
    }

    fn is_under_full(&self) -> bool {
        self.base.count == 37
    }

    fn insert(&mut self, key: u8, node: *mut BaseNode) {
        self.children[key as usize] = node;
        self.base.count += 1;
    }

    fn change(&mut self, key: u8, val: *mut BaseNode) {
        self.children[key as usize] = val;
    }

    fn get_child(&self, key: u8) -> Option<*mut BaseNode> {
        let child = self.children[key as usize];
        if child.is_null() {
            return None;
        } else {
            return Some(child);
        }
    }

    fn get_any_child(&self) -> *const BaseNode {
        let mut any_child = std::ptr::null();

        for c in self.children.iter() {
            if !((*c).is_null()) {
                if BaseNode::is_leaf(*c) {
                    return *c;
                } else {
                    any_child = *c;
                }
            }
        }
        return any_child;
    }
}
