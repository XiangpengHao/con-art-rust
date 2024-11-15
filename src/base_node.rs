#[cfg(all(feature = "shuttle", test))]
use shuttle::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(all(feature = "shuttle", test)))]
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_epoch::Guard;

use crate::{
    error::ArtError,
    lock::{ConcreteReadGuard, ReadGuard},
    node_16::{Node16, Node16Iter},
    node_256::{Node256, Node256Iter},
    node_4::{Node4, Node4Iter},
    node_48::{Node48, Node48Iter},
    node_ptr::NodePtr,
    Allocator,
};

pub(crate) const MAX_KEY_LEN: usize = 8;

/// Prefix has to be 8 bytes for better alignment.
const MAX_PREFIX_LEN: usize = 8;
pub(crate) type Prefix = [u8; MAX_PREFIX_LEN];

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NodeType {
    N4 = 0,
    N16 = 1,
    N48 = 2,
    N256 = 3,
}

impl NodeType {
    pub(crate) fn node_layout(&self) -> std::alloc::Layout {
        match *self {
            NodeType::N4 => std::alloc::Layout::from_size_align(
                std::mem::size_of::<Node4>(),
                std::mem::align_of::<Node4>(),
            )
            .unwrap(),
            NodeType::N16 => std::alloc::Layout::from_size_align(
                std::mem::size_of::<Node16>(),
                std::mem::align_of::<Node16>(),
            )
            .unwrap(),
            NodeType::N48 => std::alloc::Layout::from_size_align(
                std::mem::size_of::<Node48>(),
                std::mem::align_of::<Node48>(),
            )
            .unwrap(),
            NodeType::N256 => std::alloc::Layout::from_size_align(
                std::mem::size_of::<Node256>(),
                std::mem::align_of::<Node256>(),
            )
            .unwrap(),
        }
    }
}

pub(crate) trait Node {
    fn base(&self) -> &BaseNode;
    fn is_full(&self) -> bool;
    fn insert(&mut self, key: u8, node: NodePtr);
    fn change(&mut self, key: u8, val: NodePtr) -> NodePtr;
    fn get_child(&self, key: u8) -> Option<NodePtr>;
    fn get_children(&self, start: u8, end: u8) -> NodeIter<'_>;
    fn remove(&mut self, k: u8);
    fn copy_to<N: Node>(&self, dst: &mut N);
    fn get_type() -> NodeType;

    #[cfg(feature = "db_extension")]
    fn get_random_child(&self, rng: &mut impl rand::Rng) -> Option<(u8, NodePtr)>;
}

pub(crate) enum NodeIter<'a> {
    N4(Node4Iter<'a>),
    N16(Node16Iter<'a>),
    N48(Node48Iter<'a>),
    N256(Node256Iter<'a>),
}

impl Iterator for NodeIter<'_> {
    type Item = (u8, NodePtr);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NodeIter::N4(iter) => iter.next(),
            NodeIter::N16(iter) => iter.next(),
            NodeIter::N48(iter) => iter.next(),
            NodeIter::N256(iter) => iter.next(),
        }
    }
}

#[repr(C)]
pub(crate) struct BaseNode {
    // 2b type | 60b version | 1b lock | 1b obsolete
    pub(crate) type_version_lock_obsolete: AtomicUsize,
    pub(crate) meta: NodeMeta,
}

pub(crate) struct NodeMeta {
    prefix_cnt: u32,
    pub(crate) count: u16,
    node_type: NodeType,
    prefix: Prefix,
}

#[cfg(not(feature = "shuttle"))]
mod layout_assertion {
    use super::*;
    const _: () = assert!(std::mem::size_of::<NodeMeta>() == 16);
    const _: () = assert!(std::mem::align_of::<NodeMeta>() == 4);
    const _: () = assert!(std::mem::size_of::<BaseNode>() == 24);
    const _: () = assert!(std::mem::align_of::<BaseNode>() == 8);
}

macro_rules! gen_method {
    ($method_name:ident, ($($arg_n:ident : $args:ty),*), $return:ty) => {
        impl BaseNode {
            pub(crate) fn $method_name(&self, $($arg_n : $args),*) -> $return {
                match self.get_type() {
                    NodeType::N4 => {
                        let node = unsafe{&* (self as *const BaseNode as *const Node4)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N16 => {
                        let node = unsafe{&* (self as *const BaseNode as *const Node16)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N48 => {
                        let node = unsafe{&* (self as *const BaseNode as *const Node48)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N256 => {
                        let node = unsafe{&* (self as *const BaseNode as *const Node256)};
                        node.$method_name($($arg_n),*)
                    },
                }
            }
        }
    };
}

macro_rules! gen_method_mut {
    ($method_name:ident, ($($arg_n:ident : $args:ty),*), $return:ty) => {
        impl BaseNode {
            pub(crate) fn $method_name(&mut self, $($arg_n : $args),*) -> $return {
                match self.get_type() {
                    NodeType::N4 => {
                        let node = unsafe{&mut * (self as *mut BaseNode as *mut Node4)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N16 => {
                        let node = unsafe{&mut * (self as *mut BaseNode as *mut Node16)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N48 => {
                        let node = unsafe{&mut * (self as *mut BaseNode as *mut Node48)};
                        node.$method_name($($arg_n),*)
                    },
                    NodeType::N256 => {
                        let node = unsafe{&mut * (self as *mut BaseNode as *mut Node256)};
                        node.$method_name($($arg_n),*)
                    },
                }
            }
        }
    };
}

gen_method!(get_child, (k: u8), Option<NodePtr>);
gen_method!(get_children, (start: u8, end: u8), NodeIter<'_>);

#[cfg(feature = "db_extension")]
gen_method!(
    get_random_child,
    (rng: &mut impl rand::Rng),
    Option<(u8, NodePtr)>
);
gen_method_mut!(change, (key: u8, val: NodePtr), NodePtr);
gen_method_mut!(remove, (key: u8), ());

impl BaseNode {
    pub(crate) fn new(n_type: NodeType, prefix: &[u8]) -> Self {
        let mut prefix_v: [u8; MAX_KEY_LEN] = [0; MAX_KEY_LEN];

        assert!(prefix.len() <= MAX_KEY_LEN);
        for (i, v) in prefix.iter().enumerate() {
            prefix_v[i] = *v;
        }

        let meta = NodeMeta {
            prefix_cnt: prefix.len() as u32,
            count: 0,
            prefix: prefix_v,
            node_type: n_type,
        };

        BaseNode {
            type_version_lock_obsolete: AtomicUsize::new(0),
            meta,
        }
    }

    pub(crate) fn make_node<N: Node>(
        prefix: &[u8],
        allocator: &impl Allocator,
    ) -> Result<*mut N, ArtError> {
        let layout = N::get_type().node_layout();
        let ptr = allocator
            .allocate_zeroed(layout)
            .map_err(|_e| ArtError::Oom)?;
        let ptr = ptr.as_ptr() as *mut BaseNode;
        let node = BaseNode::new(N::get_type(), prefix);
        unsafe {
            std::ptr::write(ptr, node);

            if matches!(N::get_type(), NodeType::N48) {
                let mem = ptr as *mut Node48;
                (*mem).init_empty();
            }

            Ok(ptr as *mut N)
        }
    }

    /// Here we must get a clone of allocator because the drop_node might be called in epoch guard
    pub(crate) unsafe fn drop_node<A: Allocator>(node: *mut BaseNode, allocator: A) {
        let layout = (*node).get_type().node_layout();
        let ptr = std::ptr::NonNull::new(node as *mut u8).unwrap();
        allocator.deallocate(ptr, layout);
    }

    pub(crate) fn get_type(&self) -> NodeType {
        self.meta.node_type
    }

    #[inline]
    pub(crate) fn read_lock(&self) -> Result<ReadGuard, ArtError> {
        let version = self.type_version_lock_obsolete.load(Ordering::Acquire);

        if Self::is_locked(version) || Self::is_obsolete(version) {
            return Err(ArtError::Locked);
        }

        Ok(ReadGuard::new(version, self))
    }

    fn is_locked(version: usize) -> bool {
        (version & 0b10) == 0b10
    }

    fn is_obsolete(version: usize) -> bool {
        (version & 1) == 1
    }

    pub(crate) fn get_count(&self) -> usize {
        self.meta.count as usize
    }

    pub(crate) fn prefix_matches(&self, key: &[u8]) -> bool {
        key.starts_with(self.prefix())
    }

    pub(crate) fn prefix(&self) -> &[u8] {
        unsafe {
            self.meta
                .prefix
                .get_unchecked(..self.meta.prefix_cnt as usize)
        }
    }

    pub(crate) fn prefix_len(&self) -> usize {
        self.meta.prefix_cnt as usize
    }

    pub(crate) fn insert_grow<CurT: Node, BiggerT: Node, A: Allocator + Send + Clone + 'static>(
        n: ConcreteReadGuard<CurT>,
        parent: (u8, Option<ReadGuard>),
        val: (u8, NodePtr),
        allocator: &A,
        guard: &Guard,
    ) -> Result<(), ArtError> {
        if !n.as_ref().is_full() {
            if let Some(p) = parent.1 {
                p.unlock()?;
            }

            let mut write_n = n.upgrade().map_err(|v| v.1)?;

            write_n.as_mut().insert(val.0, val.1);
            return Ok(());
        }

        let p = parent
            .1
            .expect("parent node must present when current node is full");

        let mut write_p = p.upgrade().map_err(|v| v.1)?;

        let mut write_n = n.upgrade().map_err(|v| v.1)?;

        let n_big = BaseNode::make_node::<BiggerT>(write_n.as_ref().base().prefix(), allocator)?;
        write_n.as_ref().copy_to(unsafe { &mut *n_big });
        unsafe { &mut *n_big }.insert(val.0, val.1);

        write_p
            .as_mut()
            .change(parent.0, NodePtr::from_node(n_big as *mut BaseNode));

        write_n.mark_obsolete();
        let delete_n = write_n.as_mut() as *mut CurT as usize;
        std::mem::forget(write_n);
        let allocator: A = allocator.clone();
        guard.defer(move || unsafe {
            BaseNode::drop_node(delete_n as *mut BaseNode, allocator);
        });
        Ok(())
    }

    pub(crate) fn insert_and_unlock<'a, A: Allocator + Send + Clone + 'static>(
        node: ReadGuard<'a>,
        parent: (u8, Option<ReadGuard>),
        val: (u8, NodePtr),
        allocator: &'a A,
        guard: &Guard,
    ) -> Result<(), ArtError> {
        match node.as_ref().get_type() {
            NodeType::N4 => Self::insert_grow::<Node4, Node16, A>(
                node.into_concrete(),
                parent,
                val,
                allocator,
                guard,
            ),
            NodeType::N16 => Self::insert_grow::<Node16, Node48, A>(
                node.into_concrete(),
                parent,
                val,
                allocator,
                guard,
            ),
            NodeType::N48 => Self::insert_grow::<Node48, Node256, A>(
                node.into_concrete(),
                parent,
                val,
                allocator,
                guard,
            ),
            NodeType::N256 => Self::insert_grow::<Node256, Node256, A>(
                node.into_concrete(),
                parent,
                val,
                allocator,
                guard,
            ),
        }
    }
}
