use crate::base_node::BaseNode;

#[allow(dead_code)]
pub(crate) union NodePtr2 {
    ptr: *const BaseNode,
    payload: usize,
}

/// This no longer relevant, we can delete it
#[derive(Clone, Copy)]
pub(crate) struct NodePtr {
    val: usize,
}

impl NodePtr {
    #[inline]
    pub(crate) fn from_null() -> Self {
        Self { val: 0 }
    }

    #[inline]
    pub(crate) fn from_node(ptr: *const BaseNode) -> Self {
        Self { val: ptr as usize }
    }

    #[inline]
    pub(crate) fn from_tid(tid: usize) -> Self {
        Self { val: tid }
    }

    #[inline]
    pub(crate) fn as_tid(&self) -> usize {
        self.val
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const BaseNode {
        self.val as *const BaseNode
    }
}
