use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use douhua::MemType;

use crate::{node_256::Node256, node_4::Node4, Art, CongeeAllocator};

struct SmallAllocator {
    max_size: AtomicUsize,
}

unsafe impl CongeeAllocator for Arc<SmallAllocator> {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
        mem_type: MemType,
    ) -> Result<std::ptr::NonNull<[u8]>, douhua::AllocError> {
        let current_size = self.max_size.load(Ordering::Relaxed);
        if current_size >= layout.size() {
            self.max_size
                .store(current_size - layout.size(), Ordering::Relaxed);
            let ptr = unsafe { std::alloc::alloc(layout) };
            let ptr_slice = std::ptr::slice_from_raw_parts_mut(ptr, layout.size());
            Ok(std::ptr::NonNull::new(ptr_slice).unwrap())
        } else {
            return Err(douhua::AllocError::OutOfMemory);
        }
    }

    unsafe fn deallocate(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        _mem_type: douhua::MemType,
    ) {
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}

#[should_panic]
#[test]
fn too_small_to_new() {
    let allocator = Arc::new(SmallAllocator {
        max_size: AtomicUsize::new(1024),
    });
    let _art = Art::<usize, usize, Arc<SmallAllocator>>::new(allocator.clone());
}

#[test]
fn init_but_no_insert() {
    let allocator = Arc::new(SmallAllocator {
        max_size: AtomicUsize::new(std::mem::size_of::<Node256>()),
    });

    let art = Art::<usize, usize, Arc<SmallAllocator>>::new(allocator.clone());
    let guard = art.pin();
    let rv = art.insert(100, 100, &guard);
    assert!(rv.is_err());

    let rv = art.compute_or_insert(100, |_| 100, &guard);
    assert!(rv.is_err());
}

#[test]
fn insert_but_only_once() {
    let allocator = Arc::new(SmallAllocator {
        max_size: AtomicUsize::new(std::mem::size_of::<Node256>() + std::mem::size_of::<Node4>()),
    });

    let art = Art::<usize, usize, Arc<SmallAllocator>>::new(allocator.clone());
    let guard = art.pin();
    let rv = art.insert(0, 100, &guard);
    assert!(rv.is_ok());

    let rv = art.insert(usize::MAX, 100, &guard);
    assert!(rv.is_err());
}
