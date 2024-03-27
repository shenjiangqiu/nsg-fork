#![feature(stdarch_x86_avx512)]
use std::{
    collections::VecDeque,
    ffi::{c_char, CStr},
    fs::File,
    hint::black_box,
    io::{BufReader, BufWriter},
    path::Path,
    time::Instant,
};
pub mod knn_graph;
use itertools::Itertools;
use nsg_index::IndexNsg;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use tracing::{info, level_filters::LevelFilter};

use crate::dfs::generate_dfs;
pub mod bdfs;
pub mod bfs;
pub mod cabdfs;
pub mod dfs;
pub mod distance;
pub mod fvec;
pub mod nsg_index;
pub const NUM_TRAVERSAL: usize = 8;

#[derive(Serialize, Deserialize, Debug)]
struct Report<'a> {
    time: u64,
    traversal: &'a str,
    num_thread: usize,
    old: bool,
    barrial: Option<usize>,
}
pub fn bench_build_index_sync(
    r: usize,
    l: usize,
    c: usize,
    knn_path: &Path,
    f_vec_path: &Path,
    result_path: &Path,
) {
    bench_build_index(r, l, c, knn_path, f_vec_path, result_path, true)
}
pub fn bench_build_index(
    r: usize,
    l: usize,
    c: usize,
    knn_path: &Path,
    f_vec_path: &Path,
    result_path: &Path,
    sync: bool,
) {
    let knn_graph = knn_graph::KnnGraph::from_file(knn_path);
    let fvec = fvec::Fvec::from_file(f_vec_path);
    let index_nsg = IndexNsg::new(&knn_graph, &fvec, l, r, c);
    let center_point_id = index_nsg.build_center_point();
    let traversals = [
        ("sequential_cold", (0..knn_graph.num as u32).collect_vec()),
        ("sequential_hot", (0..knn_graph.num as u32).collect_vec()),
        (
            "bfs",
            bfs::generate_bfs(&knn_graph.final_graph)
                .into_iter()
                .map(|x| x as u32)
                .collect_vec(),
        ),
        (
            "dfs",
            generate_dfs(&knn_graph.final_graph)
                .into_iter()
                .map(|x| x as u32)
                .collect_vec(),
        ),
        // (
        //     "bdfs-4",
        //     bdfs::generate_bdfs(&knn_graph.final_graph, 4)
        //         .into_iter()
        //         .map(|x| x as u32)
        //         .collect_vec(),
        // ),
        (
            "bdfs-8",
            bdfs::generate_bdfs(&knn_graph.final_graph, 8)
                .into_iter()
                .map(|x| x as u32)
                .collect_vec(),
        ),
        // (
        //     "bdfs-16",
        //     bdfs::generate_bdfs(&knn_graph.final_graph, 16)
        //         .into_iter()
        //         .map(|x| x as u32)
        //         .collect_vec(),
        // ),
    ];
    let mut all_results = vec![];
    for num_thread in [80] {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_thread)
            .build()
            .unwrap();
        for (name, traversal) in &traversals {
            // info!("runing async: {} {}", name, num_thread);
            // let mut cut_graph = vec![SimpleNeighbor::default(); knn_graph.num * r];
            // let now = Instant::now();
            // // run sequential twice
            // thread_pool.install(|| {
            //     index_nsg.build(
            //         &traversal,
            //         &mut cut_graph,
            //         center_point_id,
            //         sync,
            //         false,
            //         None,
            //     );
            // });
            // black_box(cut_graph);
            // let elapsed = now.elapsed();
            // let secs = elapsed.as_secs();
            // info!("{} {} {}", secs, name, num_thread);
            // all_results.push(Report {
            //     time: secs,
            //     traversal: name,
            //     num_thread,
            //     old: false,
            //     barrial: None,
            // });
            // run the old
            let mut cut_graph = vec![SimpleNeighbor::default(); knn_graph.num * r];

            thread_pool.install(|| {
                info!("runing async: {} {}, OLD impl", name, num_thread,);
                let now = Instant::now();
                index_nsg.build(
                    &traversal,
                    &mut cut_graph,
                    center_point_id,
                    sync,
                    true,
                    None,
                );

                black_box(cut_graph);

                let elapsed = now.elapsed();
                let secs = elapsed.as_secs();
                info!("{} {} {}", secs, name, num_thread);
                all_results.push(Report {
                    time: secs,
                    traversal: name,
                    num_thread,
                    old: true,
                    barrial: None,
                });
            });
            for barrier in [] {
                info!(
                    "runing async: {} {}, barrier: {}",
                    name, num_thread, barrier
                );
                let mut cut_graph = vec![SimpleNeighbor::default(); knn_graph.num * r];
                let now = Instant::now();
                thread_pool.install(|| {
                    index_nsg.build(
                        &traversal,
                        &mut cut_graph,
                        center_point_id,
                        sync,
                        false,
                        Some(barrier),
                    );
                });
                black_box(cut_graph);
                let elapsed = now.elapsed();
                let secs = elapsed.as_secs();
                info!("{} {} {}", secs, name, num_thread);
                all_results.push(Report {
                    time: secs,
                    traversal: name,
                    num_thread,
                    old: false,
                    barrial: Some(barrier),
                });
            }
        }
    }
    let writer = BufWriter::new(File::create(result_path).unwrap());
    serde_json::to_writer(writer, &all_results).unwrap();
}
pub fn bench_build_index_async(
    r: usize,
    l: usize,
    c: usize,
    knn_path: &Path,
    f_vec_path: &Path,
    result_path: &Path,
) {
    bench_build_index(r, l, c, knn_path, f_vec_path, result_path, false)
}

#[no_mangle]
pub extern "C" fn rust_lib_helloworld() {
    println!("Hello, world!");
}

#[no_mangle]
pub extern "C" fn build_traversal_seqence(name: *const c_char) -> *const *const usize {
    use knn_graph::KnnGraph;
    let c_str = unsafe { CStr::from_ptr(name) };
    let file_name = Path::new(c_str.to_str().unwrap());
    let index_nsg = KnnGraph::from_file(file_name);
    let sequenctial = (0..index_nsg.final_graph.len()).collect::<Vec<_>>();
    use rayon::prelude::*;
    let bfs: Vec<usize> = bfs::generate_bfs(&index_nsg.final_graph);
    let dfs = dfs::generate_dfs(&index_nsg.final_graph);
    let bdfs_4 = bdfs::generate_bdfs(&index_nsg.final_graph, 4);
    let bdfs_8 = bdfs::generate_bdfs(&index_nsg.final_graph, 8);
    let bdfs_16 = bdfs::generate_bdfs(&index_nsg.final_graph, 16);
    let bdfs_32 = bdfs::generate_bdfs(&index_nsg.final_graph, 32);
    let bdfs_64 = bdfs::generate_bdfs(&index_nsg.final_graph, 64);
    let ret: Box<[*mut [usize]; NUM_TRAVERSAL]> = Box::new([
        Box::into_raw(sequenctial.into_boxed_slice()),
        Box::into_raw(bfs.into_boxed_slice()),
        Box::into_raw(dfs.into_boxed_slice()),
        Box::into_raw(bdfs_4.into_boxed_slice()),
        Box::into_raw(bdfs_8.into_boxed_slice()),
        Box::into_raw(bdfs_16.into_boxed_slice()),
        Box::into_raw(bdfs_32.into_boxed_slice()),
        Box::into_raw(bdfs_64.into_boxed_slice()),
    ]);
    Box::into_raw(ret) as *const *const usize
}

#[no_mangle]
pub extern "C" fn release_traversal_sequence(ptr: *const *const usize) {
    let boxed = unsafe { Box::from_raw(ptr as *mut [*mut [usize]; NUM_TRAVERSAL]) };
    for i in 0..NUM_TRAVERSAL {
        let _ = unsafe { Box::from_raw(boxed[i]) };
    }
}

#[no_mangle]
pub extern "C" fn init_logger_info() {
    init_logger(LevelFilter::INFO);
}
#[no_mangle]
pub extern "C" fn init_logger_debug() {
    init_logger(LevelFilter::DEBUG);
}
#[no_mangle]
pub extern "C" fn init_logger_trace() {
    init_logger(LevelFilter::TRACE);
}

#[no_mangle]
pub extern "C" fn info(message: *const c_char) {
    log(LevelFilter::INFO, message);
}
#[no_mangle]
pub extern "C" fn debug(message: *const c_char) {
    log(LevelFilter::DEBUG, message);
}
#[no_mangle]
pub extern "C" fn trace(message: *const c_char) {
    log(LevelFilter::TRACE, message);
}
#[no_mangle]
pub extern "C" fn error(message: *const c_char) {
    log(LevelFilter::ERROR, message);
}
#[no_mangle]
pub extern "C" fn warn(message: *const c_char) {
    log(LevelFilter::WARN, message);
}

fn log(level: LevelFilter, message: *const c_char) {
    if message.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(message) };
    let r_str = c_str.to_str().unwrap();
    match level {
        LevelFilter::INFO => tracing::info!("{}", r_str),
        LevelFilter::DEBUG => tracing::debug!("{}", r_str),
        LevelFilter::TRACE => tracing::trace!("{}", r_str),
        _ => {}
    }
}

fn init_logger(level: LevelFilter) {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(level.into())
                .from_env_lossy(),
        )
        .init();
}

// unsigned id;
// float distance;
// bool flag;
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    id: u32,
    distance: f32,
    flag: bool,
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Default)]
pub struct SimpleNeighbor {
    id: u32,
    distance: f32,
}

pub fn read_neigbors(n: usize, location: &Path) -> Vec<Vec<Neighbor>> {
    let mut all_nodes: Vec<Vec<Neighbor>> = vec![];
    for i in 0..n {
        let mut neighbors = vec![];
        let file_name = format!("{}.txt", i);
        let file_path = location.join(file_name);
        let content = std::fs::read_to_string(file_path).unwrap();
        for line in content.lines() {
            let line = line.trim();
            if line.len() == 0 {
                break;
            }
            let mut parts = line.split_whitespace();
            let id = parts.next().unwrap().parse::<u32>().unwrap();
            let distance = parts.next().unwrap().parse::<f32>().unwrap();
            let neighbor = Neighbor {
                id,
                distance,
                flag: true,
            };
            neighbors.push(neighbor);
        }
        all_nodes.push(neighbors);
    }
    all_nodes
}

pub fn translate_neighbors(neighbors: &Vec<Vec<Neighbor>>) {
    let writer = BufWriter::new(File::create("neighbors.bin").unwrap());
    bincode::serialize_into(writer, neighbors).unwrap();
}

pub fn read_bin(path: &Path) -> Vec<Vec<Neighbor>> {
    let reader = File::open(path).unwrap();
    let reader = BufReader::new(reader);
    bincode::deserialize_from(reader).unwrap()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AnalyzeResult {
    pub total_tested: usize,
    pub total_shared: usize,
}
fn analyze_by_id(
    neighbors: &Vec<Vec<Neighbor>>,
    ids: impl IntoIterator<Item = usize> + Send + Sync,
    window_size: usize,
) -> AnalyzeResult {
    let mut total_tested = 0;
    let mut total_shared = 0;
    let mut cache = VecDeque::new();
    ids.into_iter().for_each(|i| {
        let i: &Vec<Neighbor> = &neighbors[i];
        total_tested += i.len();
        let cache_merged = cache
            .iter()
            .fold(std::collections::BTreeSet::new(), |acc, x| {
                acc.union(x).cloned().collect()
            });
        let i_set = i
            .iter()
            .map(|n| n.id)
            .collect::<std::collections::BTreeSet<_>>();
        let shared = i_set.intersection(&cache_merged).count();
        total_shared += shared;

        // maintain the cache
        cache.push_back(i_set);
        if cache.len() > window_size {
            cache.pop_front();
        }
    });
    AnalyzeResult {
        total_tested,
        total_shared,
    }
}

pub fn analyze_sequential(neighbors: &Vec<Vec<Neighbor>>, window_size: usize) -> AnalyzeResult {
    analyze_by_id(neighbors, 0..neighbors.len(), window_size)
}
pub fn analyze_knn_bfs(
    neighbors: &Vec<Vec<Neighbor>>,
    knn_graph: &Vec<Vec<u32>>,
    window_size: usize,
) -> AnalyzeResult {
    assert_eq!(neighbors.len(), knn_graph.len());
    let nodes_to_visite = bfs::generate_bfs(knn_graph);
    analyze_by_id(neighbors, nodes_to_visite, window_size)
}

//dfs
pub fn analyze_knn_dfs(
    neighbors: &Vec<Vec<Neighbor>>,
    knn_graph: &Vec<Vec<u32>>,
    window_size: usize,
) -> AnalyzeResult {
    assert_eq!(neighbors.len(), knn_graph.len());
    let nodes_to_visite = generate_dfs(knn_graph);
    analyze_by_id(neighbors, nodes_to_visite, window_size)
}

pub fn analyze_knn_bdfs(
    neighbors: &Vec<Vec<Neighbor>>,
    knn_graph: &Vec<Vec<u32>>,
    window_size: usize,
    max_depth: usize,
) -> AnalyzeResult {
    assert_eq!(neighbors.len(), knn_graph.len());
    let nodes_to_visite = bdfs::generate_bdfs(knn_graph, max_depth);
    analyze_by_id(neighbors, nodes_to_visite, window_size)
}

#[cfg(test)]
mod tests {}
