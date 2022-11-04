use congee::Art;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use shumai::{config, ShumaiBench};
use std::fmt::Display;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Clone, Copy, Debug, Deserialize)]
pub enum Workload {
    ReadOnly,
    InsertOnly,
    ScanOnly,
    UpdateOnly,
}

impl Display for Workload {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Serialize, Clone, Copy, Debug, Deserialize)]
pub enum IndexType {
    Flurry,
    ART,
}

impl Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
#[config(path = "bench/benchmark.toml")]
pub struct Basic {
    pub name: String,
    pub threads: Vec<usize>,
    pub time: usize,
    #[matrix]
    pub workload: Workload,
    #[matrix]
    pub index_type: IndexType,
}

struct TestBench<Index: DBIndex> {
    index: Index,
    initial_cnt: usize,
}

trait DBIndex: Send + Sync {
    type Guard<'a>
    where
        Self: 'a;

    fn pin<'a>(&'a self) -> Self::Guard<'a>;
    fn insert<'a>(&'a self, key: usize, v: usize, guard: &Self::Guard<'a>);
    fn get<'a>(&'a self, key: &usize, guard: &Self::Guard<'a>) -> Option<usize>;
    fn update<'a>(
        &'a self,
        key: &usize,
        new: usize,
        guard: &Self::Guard<'a>,
    ) -> Option<(usize, Option<usize>)>;

    fn scan<'a>(
        &'a self,
        low_key: &usize,
        high_key: &usize,
        results: &mut [(usize, usize)],
        guard: &Self::Guard<'a>,
    ) -> usize;
}

impl DBIndex for Art<usize, usize> {
    type Guard<'a> = crossbeam_epoch::Guard;

    fn pin(&self) -> Self::Guard<'_> {
        self.pin()
    }

    fn insert(&self, key: usize, v: usize, guard: &Self::Guard<'_>) {
        self.insert(key, v, guard).unwrap();
    }

    fn get(&self, key: &usize, guard: &Self::Guard<'_>) -> Option<usize> {
        self.get(key, guard)
    }

    fn update<'a>(
        &'a self,
        key: &usize,
        new: usize,
        guard: &Self::Guard<'a>,
    ) -> Option<(usize, Option<usize>)> {
        self.compute_if_present(key, |_v| Some(new), guard)
    }

    fn scan<'a>(
        &'a self,
        low_key: &usize,
        high_key: &usize,
        results: &mut [(usize, usize)],
        guard: &Self::Guard<'a>,
    ) -> usize {
        self.range(low_key, high_key, results, guard)
    }
}

impl DBIndex for flurry::HashMap<usize, usize, fxhash::FxBuildHasher> {
    type Guard<'a> = flurry::Guard<'a>;

    fn pin<'a>(&'a self) -> Self::Guard<'a> {
        self.guard()
    }

    fn insert<'a>(&self, key: usize, v: usize, guard: &Self::Guard<'a>) {
        self.insert(key, v, &guard);
    }

    fn get<'a>(&self, key: &usize, guard: &Self::Guard<'a>) -> Option<usize> {
        self.get(key, &guard).map(|v| *v)
    }

    fn update<'a>(
        &'a self,
        key: &usize,
        new: usize,
        guard: &Self::Guard<'a>,
    ) -> Option<(usize, Option<usize>)> {
        let val = self.compute_if_present(key, |_k, _v| Some(new), guard)?;
        Some((*val, Some(*val)))
    }

    fn scan<'a>(
        &'a self,
        _low_key: &usize,
        _high_key: &usize,
        _results: &mut [(usize, usize)],
        _guard: &Self::Guard<'a>,
    ) -> usize {
        unimplemented!("Flurry can't scan")
    }
}

impl<Index: DBIndex> ShumaiBench for TestBench<Index> {
    type Config = Basic;
    type Result = usize;

    fn load(&mut self) -> Option<serde_json::Value> {
        let guard = self.index.pin();
        for i in 0..self.initial_cnt {
            self.index.insert(i, i, &guard);
        }
        None
    }

    fn run(&self, context: shumai::Context<Self::Config>) -> Self::Result {
        let mut op_cnt = 0;
        let max_scan_cnt = 50;
        let mut scan_buffer = vec![(0, 0); max_scan_cnt];
        let mut rng = thread_rng();

        context.wait_for_start();

        let guard = self.index.pin();
        while context.is_running() {
            match context.config.workload {
                Workload::ReadOnly => {
                    let val = rng.gen_range(0..self.initial_cnt);
                    let r = self.index.get(&val, &guard).unwrap();
                    assert_eq!(r, val);
                }
                Workload::InsertOnly => {
                    let val = rng.gen::<usize>();
                    self.index.insert(val, val, &guard);
                }
                Workload::UpdateOnly => {
                    let key = rng.gen_range(0..self.initial_cnt);
                    let val = rng.gen::<usize>();
                    self.index.update(&key, val, &guard);
                }

                Workload::ScanOnly => {
                    let scan_cnt = rng.gen_range(0..max_scan_cnt);
                    let low_key_v = rng.gen_range(0..(self.initial_cnt - scan_cnt * 10));
                    let high_key = low_key_v + scan_cnt * 10;

                    let scanned = self
                        .index
                        .scan(&low_key_v, &high_key, &mut scan_buffer, &guard);

                    for v in scan_buffer.iter().take(scanned) {
                        assert!(v.0 >= low_key_v);
                    }
                }
            }

            op_cnt += 1;
        }
        op_cnt
    }

    fn cleanup(&mut self) -> Option<serde_json::Value> {
        None
    }
}

fn main() {
    let config = Basic::load().expect("Failed to parse config!");
    let repeat = 3;

    for c in config.iter() {
        match c.index_type {
            IndexType::Flurry => {
                let mut test_bench = TestBench {
                    index: flurry::HashMap::with_hasher(fxhash::FxBuildHasher::default()),
                    initial_cnt: 50_000_000,
                };
                let result = shumai::run(&mut test_bench, c, repeat);
                result.write_json().unwrap();
            }
            IndexType::ART => {
                let mut test_bench = TestBench {
                    index: Art::default(),
                    initial_cnt: 50_000_000,
                };
                let result = shumai::run(&mut test_bench, c, repeat);
                result.write_json().unwrap();
            }
        }
    }
}
