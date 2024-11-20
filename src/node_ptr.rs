use std::ptr::NonNull;

use crate::{
    base_node::{BaseNode, Node},
    node_256::Node256,
};

pub(crate) struct ChildIsPayload<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl ChildIsPayload<'_> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

pub(crate) struct ChildIsSubNode<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl ChildIsSubNode<'_> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

pub(crate) enum PtrType {
    Payload(usize),
    SubNode(NonNull<BaseNode>),
}

#[derive(Clone, Copy)]
pub(crate) union NodePtr {
    payload: usize,
    sub_node: NonNull<BaseNode>,
}

impl NodePtr {
    #[inline]
    pub(crate) fn from_node(ptr: &BaseNode) -> Self {
        Self {
            sub_node: NonNull::from(ptr),
        }
    }

    pub(crate) fn from_root(ptr: NonNull<Node256>) -> Self {
        Self {
            sub_node: unsafe { std::mem::transmute::<NonNull<Node256>, NonNull<BaseNode>>(ptr) },
        }
    }

    fn from_node_new(ptr: NonNull<BaseNode>) -> Self {
        Self { sub_node: ptr }
    }

    pub(crate) fn from_payload(payload: usize) -> Self {
        Self { payload }
    }

    pub(crate) unsafe fn as_payload_unchecked(&self) -> usize {
        unsafe { self.payload }
    }

    pub(crate) fn as_ptr_safe<const MAX_LEVEL: usize>(
        &self,
        current_level: usize,
    ) -> NonNull<BaseNode> {
        debug_assert!(current_level < MAX_LEVEL);
        unsafe { self.sub_node }
    }

    pub(crate) fn downcast<const K_LEN: usize>(&self, current_level: usize) -> PtrType {
        if current_level == (K_LEN - 1) {
            PtrType::Payload(unsafe { self.as_payload_unchecked() })
        } else {
            PtrType::SubNode(unsafe { self.sub_node })
        }
    }
}

pub(crate) struct AllocatedNode<N: Node> {
    ptr: NonNull<N>,
}

impl<N: Node> AllocatedNode<N> {
    pub(crate) fn new(ptr: NonNull<N>) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_mut(&mut self) -> &mut N {
        unsafe { self.ptr.as_mut() }
    }

    pub(crate) fn into_note_ptr(self) -> NodePtr {
        let ptr = self.ptr;
        std::mem::forget(self);
        unsafe { NodePtr::from_node_new(std::mem::transmute::<NonNull<N>, NonNull<BaseNode>>(ptr)) }
    }

    pub(crate) fn into_non_null(self) -> NonNull<N> {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl<N: Node> Drop for AllocatedNode<N> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(self.ptr.as_mut());
        }
    }
}
