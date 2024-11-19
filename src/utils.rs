use crate::base_node::{BaseNode, MAX_KEY_LEN};
use crate::node_ptr::{LastLevelProof, NodePtr};
use core::cell::Cell;
use core::fmt;

const SPIN_LIMIT: u32 = 6;
const YIELD_LIMIT: u32 = 10;

/// Backoff implementation from the Crossbeam, added shuttle instrumentation
pub(crate) struct Backoff {
    step: Cell<u32>,
}

impl Backoff {
    #[inline]
    pub(crate) fn new() -> Self {
        Backoff { step: Cell::new(0) }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn reset(&self) {
        self.step.set(0);
    }

    #[inline]
    pub(crate) fn spin(&self) {
        for _ in 0..1 << self.step.get().min(SPIN_LIMIT) {
            std::hint::spin_loop();
        }

        if self.step.get() <= SPIN_LIMIT {
            self.step.set(self.step.get() + 1);
        }
        #[cfg(all(feature = "shuttle", test))]
        shuttle::thread::yield_now();
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn snooze(&self) {
        if self.step.get() <= SPIN_LIMIT {
            for _ in 0..1 << self.step.get() {
                std::hint::spin_loop();
            }
        } else {
            #[cfg(all(feature = "shuttle", test))]
            shuttle::thread::yield_now();

            #[cfg(not(all(feature = "shuttle", test)))]
            ::std::thread::yield_now();
        }

        if self.step.get() <= YIELD_LIMIT {
            self.step.set(self.step.get() + 1);
        }
    }

    #[inline]
    pub(crate) fn is_completed(&self) -> bool {
        self.step.get() > YIELD_LIMIT
    }
}

impl fmt::Debug for Backoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Backoff")
            .field("step", &self.step)
            .field("is_completed", &self.is_completed())
            .finish()
    }
}

impl Default for Backoff {
    fn default() -> Backoff {
        Backoff::new()
    }
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

        self.data[self.len] = key;
        self.len += 1;
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> u8 {
        debug_assert!(self.len > 0);

        let v = self.data[self.len - 1];
        self.len -= 1;
        v
    }

    pub(crate) fn is_last_level<const K_LEN: usize>(&self) -> Option<LastLevelProof> {
        if self.len == K_LEN {
            Some(LastLevelProof {})
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn to_usize_key(&self, _last_level_proof: &LastLevelProof) -> usize {
        let val = unsafe { *((&self.data) as *const [u8; 8] as *const usize) };
        val.swap_bytes()
    }

    pub(crate) fn get_key<const N: usize>(&self) -> [u8; N] {
        assert!(self.len == N);
        self.data[..N].try_into().unwrap()
    }

    #[inline]
    pub(crate) fn append_prefix(node: NodePtr, key_tracker: &KeyTracker) -> KeyTracker {
        let mut cur_key = key_tracker.clone();
        if key_tracker.len() == MAX_KEY_LEN {
            cur_key
        } else {
            let node_ref = BaseNode::read_lock::<MAX_KEY_LEN>(node, 0).unwrap();
            let n_prefix = node_ref.as_ref().prefix().iter().skip(key_tracker.len());
            for i in n_prefix {
                cur_key.push(*i);
            }
            cur_key
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}
