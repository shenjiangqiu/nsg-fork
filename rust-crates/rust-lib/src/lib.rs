use std::{
    collections::VecDeque,
    ffi::{c_char, CStr},
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};
pub mod knn_graph;
use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;

use crate::dfs::generate_dfs;
pub mod bdfs;
pub mod bfs;
pub mod cabdfs;
pub mod dfs;

pub const NUM_TRAVERSAL: usize = 8;
#[no_mangle]
pub extern "C" fn rust_lib_helloworld() {
    println!("Hello, world!");
}

#[no_mangle]
pub extern "C" fn build_traversal_seqence(name: *const c_char) -> *const *const usize {
    use knn_graph::IndexNSG;
    let c_str = unsafe { CStr::from_ptr(name) };
    let mut index_nsg = IndexNSG::new();
    let file_name = Path::new(c_str.to_str().unwrap());
    index_nsg.load_nn_graph(file_name).unwrap();
    let sequenctial = (0..index_nsg.final_graph_.len()).collect::<Vec<_>>();
    let bfs: Vec<usize> = bfs::generate_bfs(&index_nsg.final_graph_);
    let dfs = dfs::generate_dfs(&index_nsg.final_graph_);
    let bdfs_4 = bdfs::generate_bdfs(&index_nsg.final_graph_, 4);
    let bdfs_8 = bdfs::generate_bdfs(&index_nsg.final_graph_, 8);
    let bdfs_16 = bdfs::generate_bdfs(&index_nsg.final_graph_, 16);
    let bdfs_32 = bdfs::generate_bdfs(&index_nsg.final_graph_, 32);
    let bdfs_64 = bdfs::generate_bdfs(&index_nsg.final_graph_, 64);
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
#[derive(Serialize, Deserialize, Debug)]
pub struct Neighbor {
    id: u32,
    distance: f32,
    // flag: bool,
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
            let neighbor = Neighbor { id, distance };
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
