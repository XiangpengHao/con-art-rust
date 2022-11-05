#[cfg(all(feature = "shuttle", test))]
use shuttle::thread;

#[cfg(not(all(feature = "shuttle", test)))]
use std::thread;

use crate::key::{GeneralKey, RawKey};
use crate::tree::RawTree;
use std::sync::Arc;

#[test]
fn small_insert() {
    let key_cnt = 10_000;
    let tree = RawTree::default();

    let guard = crossbeam_epoch::pin();
    for k in 0..key_cnt {
        tree.insert(GeneralKey::key_from(k), k, &guard).unwrap();
        let v = tree.get(&GeneralKey::key_from(k), &guard).unwrap();
        assert_eq!(v, k);
    }
}

#[test]
fn test_sparse_keys() {
    let key_cnt = 100_000;
    let tree = RawTree::default();
    let mut keys = Vec::<usize>::with_capacity(key_cnt);

    let guard = crossbeam_epoch::pin();
    for _i in 0..key_cnt {
        let k = thread_rng().gen::<usize>() & 0x7fff_ffff_ffff_ffff;
        keys.push(k);

        tree.insert(GeneralKey::key_from(k), k, &guard).unwrap();
    }

    let delete_cnt = key_cnt / 2;

    for i in keys.iter().take(delete_cnt) {
        let _rt = tree
            .compute_if_present(&GeneralKey::key_from(*i), &mut |_v| None, &guard)
            .unwrap();
    }

    for i in keys.iter().take(delete_cnt) {
        let v = tree.get(&GeneralKey::key_from(*i), &guard);
        assert!(v.is_none());
    }

    for i in keys.iter().skip(delete_cnt) {
        let v = tree.get(&GeneralKey::key_from(*i), &guard).unwrap();
        assert_eq!(v, *i);
    }

    #[cfg(feature = "stats")]
    println!("{}", tree.stats());
}

use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};

#[test]
fn test_concurrent_insert() {
    let key_cnt_per_thread = 5_000;
    let n_thread = 3;
    let mut key_space = Vec::with_capacity(key_cnt_per_thread * n_thread);
    for i in 0..key_space.capacity() {
        key_space.push(i);
    }
    let mut r = StdRng::seed_from_u64(42);
    key_space.shuffle(&mut r);

    let key_space = Arc::new(key_space);

    let tree = Arc::new(RawTree::default());

    let mut handlers = Vec::new();
    for t in 0..n_thread {
        let key_space = key_space.clone();
        let tree = tree.clone();

        handlers.push(thread::spawn(move || {
            let guard = crossbeam_epoch::pin();
            for i in 0..key_cnt_per_thread {
                let idx = t * key_cnt_per_thread + i;
                let val = key_space[idx];
                tree.insert(GeneralKey::key_from(val), val, &guard).unwrap();
            }
        }));
    }

    for h in handlers.into_iter() {
        h.join().unwrap();
    }

    let guard = crossbeam_epoch::pin();
    for v in key_space.iter() {
        let val = tree.get(&GeneralKey::key_from(*v), &guard).unwrap();
        assert_eq!(val, *v);
    }
}

#[cfg(all(feature = "shuttle", test))]
#[test]
fn shuttle_insert_only() {
    let mut config = shuttle::Config::default();
    config.max_steps = shuttle::MaxSteps::None;
    let mut runner = shuttle::PortfolioRunner::new(true, config);
    runner.add(shuttle::scheduler::PctScheduler::new(5, 2_000));
    runner.add(shuttle::scheduler::PctScheduler::new(5, 2_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(2_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(2_000));

    runner.run(test_concurrent_insert);
}

#[test]
fn test_concurrent_insert_read() {
    let key_cnt_per_thread = 5_000;
    let w_thread = 2;
    let mut key_space = Vec::with_capacity(key_cnt_per_thread * w_thread);
    for i in 0..key_space.capacity() {
        key_space.push(i);
    }

    let mut r = StdRng::seed_from_u64(42);
    key_space.shuffle(&mut r);

    let key_space = Arc::new(key_space);

    let tree = Arc::new(RawTree::default());

    let mut handlers = Vec::new();
    for t in 0..w_thread {
        let key_space = key_space.clone();
        let tree = tree.clone();
        handlers.push(thread::spawn(move || {
            let guard = crossbeam_epoch::pin();
            for i in 0..key_cnt_per_thread {
                let idx = t * key_cnt_per_thread + i;
                let val = key_space[idx];
                tree.insert(GeneralKey::key_from(val), val, &guard).unwrap();
            }
        }));
    }

    let r_thread = 2;
    for t in 0..r_thread {
        let tree = tree.clone();
        handlers.push(thread::spawn(move || {
            let mut r = StdRng::seed_from_u64(10 + t);
            let guard = crossbeam_epoch::pin();
            for _i in 0..key_cnt_per_thread {
                let val = r.gen_range(0..(key_cnt_per_thread * w_thread));
                if let Some(v) = tree.get(&GeneralKey::key_from(val), &guard) {
                    assert_eq!(v, val);
                }
            }
        }));
    }

    for h in handlers.into_iter() {
        h.join().unwrap();
    }

    let guard = crossbeam_epoch::pin();
    for v in key_space.iter() {
        let val = tree.get(&GeneralKey::key_from(*v), &guard).unwrap();
        assert_eq!(val, *v);
    }
}

#[cfg(all(feature = "shuttle", test))]
#[test]
fn shuttle_concurrent_insert_read() {
    let mut config = shuttle::Config::default();
    config.max_steps = shuttle::MaxSteps::None;
    let mut runner = shuttle::PortfolioRunner::new(true, config);
    runner.add(shuttle::scheduler::PctScheduler::new(5, 1_000));
    runner.add(shuttle::scheduler::PctScheduler::new(5, 1_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(1_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(1_000));

    runner.run(test_concurrent_insert_read);
}
