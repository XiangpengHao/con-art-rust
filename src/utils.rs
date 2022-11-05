use crate::base_node::{Prefix, MAX_KEY_LEN};
use crate::key::RawKey;
use crate::node_ptr::NodePtr;
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

#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct PrefixKeysTracker {
    len: usize,
    data: Prefix,
}

impl std::hash::Hash for PrefixKeysTracker {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data[..self.len].hash(state);
    }
}

impl PrefixKeysTracker {
    pub(crate) fn from_raw_key(key: &impl RawKey, max_len: usize) -> Self {
        let mut data = [0; MAX_KEY_LEN];
        let len = std::cmp::min(max_len, key.len());

        data[..len].copy_from_slice(&key.as_bytes()[..len]);
        Self { len, data }
    }

    pub(crate) fn from_prefix(prefix: &Prefix, len: usize) -> Self {
        let mut data = [0; MAX_KEY_LEN];
        data[..len].copy_from_slice(&prefix[..len]);
        Self { len, data }
    }

    #[inline]
    pub(crate) fn push(&mut self, key: u8) {
        debug_assert!(self.len <= MAX_KEY_LEN);

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

    #[inline]
    pub(crate) fn to_usize_key(&self) -> usize {
        assert!(self.len == MAX_KEY_LEN);
        let val = unsafe { *((&self.data) as *const [u8; 8] as *const usize) };
        val.swap_bytes()
    }

    #[inline]
    pub(crate) fn append_prefix(
        node: NodePtr,
        key_tracker: &PrefixKeysTracker,
    ) -> PrefixKeysTracker {
        let mut cur_key = key_tracker.clone();
        if key_tracker.len() == MAX_KEY_LEN {
            cur_key
        } else {
            let node_ref = unsafe { &*node.as_ptr() };
            let n_prefix = node_ref.prefix().iter().skip(key_tracker.len());
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

/// Inject error at 5% of the time
#[cfg(test)]
pub(crate) fn fail_point(err: crate::error::ArtError) -> Result<(), crate::error::ArtError> {
    if random(100) < 5 {
        Err(err)
    } else {
        Ok(())
    }
}

/// Generates a random number in `0..n`, code taken from sled (https://github.com/spacejam/sled/blob/main/src/debug_delay.rs#L79)
#[cfg(test)]
fn random(n: u32) -> u32 {
    use std::num::Wrapping;

    thread_local! {
        static RNG: Cell<Wrapping<u32>> = Cell::new(Wrapping(1_406_868_647));
    }

    #[allow(clippy::cast_possible_truncation)]
    RNG.try_with(|rng| {
        // This is the 32-bit variant of Xorshift.
        //
        // Source: https://en.wikipedia.org/wiki/Xorshift
        let mut x = rng.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rng.set(x);

        // This is a fast alternative to `x % n`.
        //
        // Author: Daniel Lemire
        // Source: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        (u64::from(x.0).wrapping_mul(u64::from(n)) >> 32) as u32
    })
    .unwrap_or(0)
}
