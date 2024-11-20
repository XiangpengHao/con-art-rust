use crate::{
    base_node::{BaseNode, Node, NodeIter, NodeType},
    node_ptr::NodePtr,
};

#[repr(C)]
#[repr(align(8))] // Node 16 doesn't need to align to 64 bc it occupies 3 cache lines anyway
pub(crate) struct Node16 {
    base: BaseNode,
    children: [NodePtr; 16],
    keys: [u8; 16],
}

#[cfg(not(feature = "shuttle"))]
const _: () = assert!(std::mem::size_of::<Node16>() == 168);

#[cfg(not(feature = "shuttle"))]
const _: () = assert!(std::mem::align_of::<Node16>() == 8);

impl Node16 {
    fn flip_sign(val: u8) -> u8 {
        val ^ 128
    }

    fn get_insert_pos(&self, key: u8) -> usize {
        let flipped = Self::flip_sign(key);

        let mut pos = 0;
        while pos < self.base.meta.count {
            if self.keys[pos as usize] >= flipped {
                return pos as usize;
            }
            pos += 1;
        }
        pos as usize
    }

    fn get_child_pos(&self, key: u8) -> Option<usize> {
        // TODO: xiangpeng check this code is being auto-vectorized
        let target = Self::flip_sign(key);

        self.keys
            .iter()
            .take(self.base.meta.count as usize)
            .position(|k| *k == target)
    }
}

pub(crate) struct Node16Iter<'a> {
    node: &'a Node16,
    start_pos: usize,
    end_pos: usize,
}

impl Iterator for Node16Iter<'_> {
    type Item = (u8, NodePtr);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_pos > self.end_pos {
            return None;
        }
        let key = Node16::flip_sign(self.node.keys[self.start_pos]);
        let child = self.node.children[self.start_pos];
        self.start_pos += 1;
        Some((key, child))
    }
}

impl Node for Node16 {
    fn get_type() -> NodeType {
        NodeType::N16
    }

    fn get_children(&self, start: u8, end: u8) -> NodeIter {
        if self.base.meta.count == 0 {
            // FIXME: the node may be empty due to deletion, this is not intended, we should fix the delete logic
            return NodeIter::N16(Node16Iter {
                node: self,
                start_pos: 1,
                end_pos: 0,
            });
        }
        let start_pos = self.get_child_pos(start).unwrap_or(0);
        let end_pos = self
            .get_child_pos(end)
            .unwrap_or(self.base.meta.count as usize - 1);

        debug_assert!(end_pos < 16);

        NodeIter::N16(Node16Iter {
            node: self,
            start_pos,
            end_pos,
        })
    }

    fn remove(&mut self, k: u8) {
        let pos = self
            .get_child_pos(k)
            .expect("trying to delete a non-existing key");
        unsafe {
            std::ptr::copy(
                self.keys.as_ptr().add(pos + 1),
                self.keys.as_mut_ptr().add(pos),
                self.base.meta.count as usize - pos - 1,
            );

            std::ptr::copy(
                self.children.as_ptr().add(pos + 1),
                self.children.as_mut_ptr().add(pos),
                self.base.meta.count as usize - pos - 1,
            );
        }
        self.base.meta.count -= 1;
        debug_assert!(self.get_child(k).is_none());
    }

    fn copy_to<N: Node>(&self, dst: &mut N) {
        for i in 0..self.base.meta.count {
            dst.insert(
                Self::flip_sign(self.keys[i as usize]),
                self.children[i as usize],
            );
        }
    }

    fn base(&self) -> &BaseNode {
        &self.base
    }

    fn is_full(&self) -> bool {
        self.base.meta.count == 16
    }

    // Insert must keep keys sorted, is this necessary?
    fn insert(&mut self, key: u8, node: NodePtr) {
        let key_flipped = Self::flip_sign(key);

        let pos = self.get_insert_pos(key);

        unsafe {
            std::ptr::copy(
                self.keys.as_ptr().add(pos),
                self.keys.as_mut_ptr().add(pos + 1),
                self.base.meta.count as usize - pos,
            );

            std::ptr::copy(
                self.children.as_ptr().add(pos),
                self.children.as_mut_ptr().add(pos + 1),
                self.base.meta.count as usize - pos,
            );
        }

        self.keys[pos] = key_flipped;
        self.children[pos] = node;
        self.base.meta.count += 1;

        assert!(self.base.meta.count <= 16);
    }

    fn change(&mut self, key: u8, val: NodePtr) -> NodePtr {
        let pos = self.get_child_pos(key).unwrap();
        let old = self.children[pos];
        self.children[pos] = val;
        old
    }

    fn get_child(&self, key: u8) -> Option<NodePtr> {
        let pos = self.get_child_pos(key)?;
        let child = unsafe { self.children.get_unchecked(pos) };
        Some(*child)
    }
}
