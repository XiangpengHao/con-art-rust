use con_art_rust::Art;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use shumai::{shumai_config, ShumaiBench};
use std::fmt::Display;

#[derive(Serialize, Clone, Copy, Debug, Deserialize)]
pub enum Workload {
    ReadOnly,
    InsertOnly,
    ScanOnly,
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
#[shumai_config]
pub mod test_config {
    use super::{IndexType, Workload};

    pub struct Basic {
        pub name: String,
        pub threads: Vec<usize>,
        pub time: usize,
        #[matrix]
        pub workload: Workload,
        #[matrix]
        pub index_type: IndexType,
    }
}

struct TestBench<Index: DBIndex> {
    index: Index,
    initial_cnt: usize,
}

trait DBIndex: Send + Sync {
    type Guard;

    fn pin(&self) -> Self::Guard;
    fn insert(&self, key: usize, v: usize, guard: &Self::Guard);
    fn get(&self, key: &usize, guard: &Self::Guard) -> Option<usize>;
}

impl DBIndex for Art {
    type Guard = crossbeam_epoch::Guard;

    fn pin(&self) -> Self::Guard {
        self.pin()
    }

    fn insert(&self, key: usize, v: usize, guard: &Self::Guard) {
        self.insert(key, v, guard);
    }

    fn get(&self, key: &usize, guard: &Self::Guard) -> Option<usize> {
        self.get(key, guard)
    }
}

impl DBIndex for flurry::HashMap<usize, usize> {
    type Guard = flurry::epoch::Guard;

    fn pin(&self) -> Self::Guard {
        flurry::epoch::pin()
    }

    fn insert(&self, key: usize, v: usize, guard: &Self::Guard) {
        self.insert(key, v, guard);
    }

    fn get(&self, key: &usize, guard: &Self::Guard) -> Option<usize> {
        self.get(key, guard).map(|v| *v)
    }
}

impl<Index: DBIndex> ShumaiBench for TestBench<Index> {
    type Config = test_config::Basic;
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
                    let val = rng.gen::<usize>() & 0x7fff_ffff_ffff_ffff;
                    self.index.insert(val, val, &guard);
                }
                Workload::ScanOnly => {
                    unimplemented!()
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
    let filter = std::env::args().nth(1).unwrap_or_else(|| ".*".to_string());
    let config = test_config::Basic::load_with_filter("bench/benchmark.toml", filter)
        .expect("Failed to parse config!");
    let repeat = 3;

    for c in config.iter() {
        match c.index_type {
            IndexType::Flurry => {
                let mut test_bench = TestBench {
                    index: flurry::HashMap::new(),
                    initial_cnt: 50_000_000,
                };
                let result = shumai::run(&mut test_bench, c, repeat);
                result.write_json().unwrap();
            }
            IndexType::ART => {
                let mut test_bench = TestBench {
                    index: Art::new(),
                    initial_cnt: 50_000_000,
                };
                let result = shumai::run(&mut test_bench, c, repeat);
                result.write_json().unwrap();
            }
        }
    }
}
