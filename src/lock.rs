use std::{cell::UnsafeCell, sync::atomic::Ordering};

use crate::{
    base_node::{BaseNode, Node},
    error::ArtError,
};

pub(crate) struct ConcreteReadGuard<'a, T: Node> {
    version: usize,
    node: &'a UnsafeCell<T>,
}

impl<'a, T: Node> ConcreteReadGuard<'a, T> {
    pub(crate) fn as_ref(&self) -> &T {
        unsafe { &*self.node.get() }
    }

    pub(crate) fn upgrade(self) -> Result<ConcreteWriteGuard<'a, T>, (Self, ArtError)> {
        #[cfg(test)]
        {
            if crate::utils::fail_point(ArtError::VersionNotMatch).is_err() {
                return Err((self, ArtError::VersionNotMatch));
            };
        }

        let new_version = self.version + 0b10;
        match self
            .as_ref()
            .base()
            .type_version_lock_obsolete
            .compare_exchange_weak(
                self.version,
                new_version,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
            Ok(_) => Ok(ConcreteWriteGuard {
                node: unsafe { &mut *self.node.get() },
            }),
            Err(_v) => Err((self, ArtError::VersionNotMatch)),
        }
    }
}

pub(crate) struct ConcreteWriteGuard<'a, T: Node> {
    node: &'a mut T,
}

impl<'a, T: Node> ConcreteWriteGuard<'a, T> {
    pub(crate) fn as_ref(&self) -> &T {
        self.node
    }

    pub(crate) fn as_mut(&mut self) -> &mut T {
        self.node
    }

    pub(crate) fn mark_obsolete(&self) {
        self.node
            .base()
            .type_version_lock_obsolete
            .fetch_add(0b01, Ordering::Release);
    }
}

impl<'a, T: Node> Drop for ConcreteWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.node
            .base()
            .type_version_lock_obsolete
            .fetch_add(0b10, Ordering::Release);
    }
}

pub(crate) struct ReadGuard<'a> {
    version: usize,
    node: &'a UnsafeCell<BaseNode>,
}

impl<'a> ReadGuard<'a> {
    pub(crate) fn new(v: usize, node: &'a BaseNode) -> Self {
        Self {
            version: v,
            node: unsafe { &*(node as *const BaseNode as *const UnsafeCell<BaseNode>) }, // todo: the caller should pass UnsafeCell<BaseNode> instead
        }
    }

    pub(crate) fn check_version(&self) -> Result<usize, ArtError> {
        let v = self
            .as_ref()
            .type_version_lock_obsolete
            .load(Ordering::Acquire);

        if v == self.version {
            Ok(v)
        } else {
            Err(ArtError::VersionNotMatch)
        }
    }

    pub(crate) fn unlock(self) -> Result<usize, ArtError> {
        self.check_version()
    }

    #[must_use]
    pub(crate) fn into_concrete<T: Node>(self) -> ConcreteReadGuard<'a, T> {
        assert_eq!(self.as_ref().get_type(), T::get_type());

        ConcreteReadGuard {
            version: self.version,
            node: unsafe { &*(self.node as *const UnsafeCell<BaseNode> as *const UnsafeCell<T>) },
        }
    }

    pub(crate) fn as_ref(&self) -> &BaseNode {
        unsafe { &*self.node.get() }
    }

    pub(crate) fn upgrade(self) -> Result<WriteGuard<'a>, (Self, ArtError)> {
        #[cfg(test)]
        {
            if crate::utils::fail_point(ArtError::VersionNotMatch).is_err() {
                return Err((self, ArtError::VersionNotMatch));
            };
        }

        let new_version = self.version + 0b10;
        match self
            .as_ref()
            .type_version_lock_obsolete
            .compare_exchange_weak(
                self.version,
                new_version,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
            Ok(_) => Ok(WriteGuard {
                node: unsafe { &mut *self.node.get() },
            }),
            Err(_v) => Err((self, ArtError::VersionNotMatch)),
        }
    }
}

pub(crate) struct WriteGuard<'a> {
    node: &'a mut BaseNode,
}

impl<'a> WriteGuard<'a> {
    pub(crate) fn as_ref(&self) -> &BaseNode {
        self.node
    }

    pub(crate) fn as_mut(&mut self) -> &mut BaseNode {
        self.node
    }

    pub(crate) fn mark_obsolete(&mut self) {
        self.node
            .type_version_lock_obsolete
            .fetch_add(0b01, Ordering::Release);
    }
}

impl<'a> Drop for WriteGuard<'a> {
    fn drop(&mut self) {
        self.node
            .type_version_lock_obsolete
            .fetch_add(0b10, Ordering::Release);
    }
}
