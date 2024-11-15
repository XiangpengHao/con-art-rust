use crate::{
    base_node::{BaseNode, Node, NodeIter, NodeType},
    node_ptr::NodePtr,
};

#[repr(C)]
#[repr(align(8))]
pub(crate) struct Node256 {
    base: BaseNode,
    key_mask: [u8; 32],
    children: [NodePtr; 256],
}

#[cfg(not(feature = "shuttle"))]
const _: () = assert!(std::mem::size_of::<Node256>() == 2104);
#[cfg(not(feature = "shuttle"))]
const _: () = assert!(std::mem::align_of::<Node256>() == 8);

impl Node256 {
    #[inline]
    fn set_mask(&mut self, key: usize) {
        let idx = key / 8;
        let bit = key % 8;
        self.key_mask[idx] |= 1 << bit;
    }

    #[inline]
    fn unset_mask(&mut self, key: usize) {
        let idx = key / 8;
        let bit = key % 8;
        self.key_mask[idx] &= !(1 << bit);
    }

    #[inline]
    fn get_mask(&self, key: usize) -> bool {
        let idx = key / 8;
        let bit = key % 8;
        let key_mask = unsafe { self.key_mask.get_unchecked(idx) };
        *key_mask & (1 << bit) != 0
    }
}

pub(crate) struct Node256Iter<'a> {
    start: u8,
    end: u8,
    idx: u16,
    node: &'a Node256,
}

impl<'a> Iterator for Node256Iter<'a> {
    type Item = (u8, NodePtr);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let cur = self.idx + self.start as u16;

            if cur > self.end as u16 {
                return None;
            }

            self.idx += 1;

            if self.node.get_mask(cur as usize) {
                return Some((cur as u8, self.node.children[cur as usize]));
            } else {
                continue;
            }
        }
    }
}

impl Node for Node256 {
    fn get_type() -> NodeType {
        NodeType::N256
    }

    fn get_children(&self, start: u8, end: u8) -> NodeIter {
        NodeIter::N256(Node256Iter {
            start,
            end,
            idx: 0,
            node: self,
        })
    }

    fn copy_to<N: Node>(&self, dst: &mut N) {
        for (i, c) in self.children.iter().enumerate() {
            if self.get_mask(i) {
                dst.insert(i as u8, *c);
            }
        }
    }

    #[inline(always)]
    fn base(&self) -> &BaseNode {
        &self.base
    }

    fn is_full(&self) -> bool {
        false
    }

    fn insert(&mut self, key: u8, node: NodePtr) {
        self.children[key as usize] = node;
        self.set_mask(key as usize);
        self.base.meta.count += 1;
    }

    fn change(&mut self, key: u8, val: NodePtr) -> NodePtr {
        let old = self.children[key as usize];
        self.children[key as usize] = val;
        old
    }

    fn remove(&mut self, k: u8) {
        self.unset_mask(k as usize);
        self.base.meta.count -= 1;
    }

    fn get_child(&self, key: u8) -> Option<NodePtr> {
        if self.get_mask(key as usize) {
            let child = unsafe { self.children.get_unchecked(key as usize) };
            Some(*child)
        } else {
            None
        }
    }

    #[cfg(feature = "db_extension")]
    fn get_random_child(&self, rng: &mut impl rand::Rng) -> Option<(u8, NodePtr)> {
        if self.base.meta.count == 0 {
            return None;
        }
        let mut scan_cnt = rng.gen_range(1..=self.base.meta.count);
        let mut idx = 0;
        while scan_cnt > 0 {
            if self.get_mask(idx) {
                scan_cnt -= 1;
            }
            idx += 1;
        }
        Some(((idx - 1) as u8, self.children[idx - 1]))
    }
}
