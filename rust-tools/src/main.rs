use rust_lib::init_logger_info;
use std::{fs::File, path::Path};
use tracing::info;

fn main() {
    init_logger_info();
    let nsg_trace_path = Path::new("neighbors.bin");
    let knn_path = Path::new("gist.100nn.graph");
    let mut knn_graph = rust_lib::knn_graph::IndexNSG::new();
    info!("Loading knn graph");
    knn_graph.load_nn_graph(knn_path).unwrap();
    info!("Loading neighbors");
    let neighbors = rust_lib::read_bin(nsg_trace_path);
    let sequential_result = rust_lib::analyze_sequential(&neighbors);
    let bfs_result = rust_lib::analyze_knn_bfs(&neighbors, &knn_graph.final_graph_);
    let dfs_result = rust_lib::analyze_knn_dfs(&neighbors, &knn_graph.final_graph_);
    let sequential_file = "./seq.json";
    serde_json::to_writer(File::create(sequential_file).unwrap(), &sequential_result).unwrap();
    let bfs_file = "./bfs.json";
    serde_json::to_writer(File::create(bfs_file).unwrap(), &bfs_result).unwrap();
    let dfs_file = "./dfs.json";
    serde_json::to_writer(File::create(dfs_file).unwrap(), &dfs_result).unwrap();
}

// fn translate() {
//     let test_path = Path::new("./test1");
//     let result = rust_lib::read_neigbors(1000000, test_path);
//     rust_lib::translate_neighbors(&result);
// }
