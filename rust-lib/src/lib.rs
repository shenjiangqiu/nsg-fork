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

#[no_mangle]
pub extern "C" fn rust_lib_helloworld() {
    println!("Hello, world!");
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
) -> AnalyzeResult {
    use itertools::Itertools;
    use rayon::prelude::*;
    let ids: Vec<(_, _)> = ids.into_iter().tuple_windows().collect();
    let (total_tested, total_shared) = ids
        .into_par_iter()
        .map(|(i, j)| {
            let i: &Vec<Neighbor> = &neighbors[i];
            let j: &Vec<Neighbor> = &neighbors[j];
            // assert_eq!(i.len(), j.len());
            let tested = i.len();
            let i_set = i
                .iter()
                .map(|n| n.id)
                .collect::<std::collections::BTreeSet<_>>();
            let j_set = j
                .iter()
                .map(|n| n.id)
                .collect::<std::collections::BTreeSet<_>>();
            let shared = i_set.intersection(&j_set).count();
            let shared = shared;
            (tested, shared)
        })
        .reduce(|| (0, 0), |(t1, s1), (t2, s2)| (t1 + t2, s1 + s2));
    AnalyzeResult {
        total_tested,
        total_shared,
    }
}

pub fn analyze_sequential(neighbors: &Vec<Vec<Neighbor>>) -> AnalyzeResult {
    analyze_by_id(neighbors, 0..neighbors.len())
}
pub fn analyze_knn_bfs(neighbors: &Vec<Vec<Neighbor>>, knn_graph: &Vec<Vec<u32>>) -> AnalyzeResult {
    assert_eq!(neighbors.len(), knn_graph.len());
    let nodes_to_visite = generate_bfs(knn_graph);
    analyze_by_id(neighbors, nodes_to_visite)
}

pub fn generate_bfs(knn_graph: &Vec<Vec<u32>>) -> Vec<usize> {
    let total_nodes = knn_graph.len();
    let mut visited_nodes_count = 0;
    let mut visited = vec![false; knn_graph.len()];
    let mut nodes_to_visite = vec![];
    let mut working_queue = VecDeque::new();
    working_queue.push_back(0);
    visited[0] = true;
    visited_nodes_count += 1;
    loop {
        while let Some(node) = working_queue.pop_front() {
            //
            nodes_to_visite.push(node);

            let neighbors = knn_graph[node]
                .iter()
                .map(|n| *n as usize)
                .collect::<Vec<_>>();
            for neighbor in neighbors {
                if visited[neighbor] {
                    continue;
                }
                working_queue.push_back(neighbor);
                visited[neighbor] = true;
                visited_nodes_count += 1;
            }
        }
        if visited_nodes_count == total_nodes {
            break;
        }
        // find a node not visited
        let unvisited = visited.iter().position(|&v| !v).unwrap();
        working_queue.push_back(unvisited);
        visited[unvisited] = true;
        visited_nodes_count += 1;
    }
    nodes_to_visite
}
//dfs
pub fn analyze_knn_dfs(neighbors: &Vec<Vec<Neighbor>>, knn_graph: &Vec<Vec<u32>>) -> AnalyzeResult {
    assert_eq!(neighbors.len(), knn_graph.len());
    let nodes_to_visite = generate_dfs(knn_graph);
    analyze_by_id(neighbors, nodes_to_visite)
}

pub fn generate_dfs(knn_graph: &Vec<Vec<u32>>) -> Vec<usize> {
    let total_nodes = knn_graph.len();
    let mut visited_nodes_count = 0;
    let mut visited = vec![false; knn_graph.len()];
    let mut nodes_to_visite = vec![];
    let mut working_stack = vec![0];
    visited[0] = true;
    visited_nodes_count += 1;
    loop {
        while let Some(node) = working_stack.pop() {
            //
            nodes_to_visite.push(node);

            let neighbors = knn_graph[node]
                .iter()
                .map(|n| *n as usize)
                .collect::<Vec<_>>();
            for neighbor in neighbors {
                if visited[neighbor] {
                    continue;
                }
                working_stack.push(neighbor);
                visited[neighbor] = true;
                visited_nodes_count += 1;
            }
        }
        if visited_nodes_count == total_nodes {
            break;
        }
        // find a node not visited
        let unvisited = visited.iter().position(|&v| !v).unwrap();
        working_stack.push(unvisited);
        visited[unvisited] = true;
        visited_nodes_count += 1;
    }
    nodes_to_visite
}

#[no_mangle]
pub extern "C" fn run_all() {}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bfs() {
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_bfs(&graph);
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6]);

        // with ilands
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6, 8],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_bfs(&graph);
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 8, 7]);
    }

    #[test]
    fn test_dfs() {
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_dfs(&graph);
        assert_eq!(result, vec![0, 2, 6, 5, 1, 4, 3]);

        // with ilands
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6, 8],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_dfs(&graph);
        assert_eq!(result, vec![0, 2, 8, 6, 5, 1, 4, 3, 7]);
    }
}
