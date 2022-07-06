# Congee 
[![congee](https://github.com/XiangpengHao/congee/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/congee/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/congee.svg)](
https://crates.io/crates/congee)
[![dependency status](https://deps.rs/repo/github/xiangpenghao/congee/status.svg)](https://deps.rs/crate/congee)
[![codecov](https://codecov.io/gh/XiangpengHao/congee/branch/main/graph/badge.svg?token=x0PSjQrqyR)](https://codecov.io/gh/XiangpengHao/congee)
[![Documentation](https://docs.rs/congee/badge.svg)](https://docs.rs/congee)

A Rust implementation of ART-OLC [concurrent adaptive radix tree](https://db.in.tum.de/~leis/papers/artsync.pdf).
It implements the optimistic lock coupling with proper SIMD support.

It only supports (and is optimized for) 8 byte key;
due to this specialization, congee has great performance -- basic operations are faster than most hash tables, range scan is an order of magnitude faster.

The code is extensively tested with [{address|leak} sanitizer](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html) as well as [libfuzzer](https://llvm.org/docs/LibFuzzer.html).

### Why Congee?
- Fast performance, faster than most hash tables.
- Concurrent, super scalable, it reaches 150Mop/s on 32 cores.
- Super low memory consumption. Hash tables often have exponential bucket size growth, which often lead to low load factors. ART is more space efficient.


### Why not Congee?
- Not for arbitrary key size. This library only supports 8 byte key.


### Example:
```rust
use congee::Art;
let art = Art::new();
let guard = art.pin(); // enter an epoch

art.insert(0, 42, &guard); // insert a value
let val = art.get(&0, &guard).unwrap(); // read the value
assert_eq!(val, 42);

let mut scan_buffer = vec![(0, 0); 8];
let scan_result = art.range(&0, &10, &mut scan_buffer, &guard); // scan values
assert_eq!(scan_result, 1);
assert_eq!(scan_buffer[0], (0, 42));
```

### History
Congee was originally developed in the [Alchemy](https://github.com/XiangpengHao/alchemy) project to fullfil its need for a concurrent, scalable, and low memory footprint range index.
