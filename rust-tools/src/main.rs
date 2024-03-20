use clap::{Parser, Subcommand};
use rust_lib::{init_logger_info, AnalyzeResult};
use std::{fs::File, path::Path};
use tracing::info;
#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// translate the txt trace into bin
    Translate,
    /// analyze the trace
    Analyze { start: usize, end: usize },
    /// analyze the result
    ParseResult { start: usize, end: usize },
}

fn main() {
    init_logger_info();
    let cli = Cli::parse();
    match cli.command {
        Commands::Translate => translate(),
        Commands::Analyze { start, end } => analyze(start, end),
        Commands::ParseResult { start, end } => parse_result(start, end),
    }
}
fn parse_result(start: usize, end: usize) {
    for i in start..end {
        let seq = format!("./s{}-seq.json", i);
        let bfs = format!("./s{}-bfs.json", i);
        let dfs = format!("./s{}-dfs.json", i);
        let seq_result: AnalyzeResult = serde_json::from_reader(File::open(seq).unwrap()).unwrap();
        let bfs_result: AnalyzeResult = serde_json::from_reader(File::open(bfs).unwrap()).unwrap();
        let dfs_result: AnalyzeResult = serde_json::from_reader(File::open(dfs).unwrap()).unwrap();
        println!(
            "missrate:{}: seq: {} bfs: {} dfs: {}",
            i,
            1. - seq_result.total_shared as f32 / seq_result.total_tested as f32,
            1. - bfs_result.total_shared as f32 / bfs_result.total_tested as f32,
            1. - dfs_result.total_shared as f32 / dfs_result.total_tested as f32
        )
    }
}
fn analyze(start: usize, end: usize) {
    let nsg_trace_path = Path::new("neighbors.bin");
    let knn_path = Path::new("gist.100nn.graph");
    let mut knn_graph = rust_lib::knn_graph::IndexNSG::new();
    info!("Loading knn graph");
    knn_graph.load_nn_graph(knn_path).unwrap();
    info!("Loading neighbors");
    let neighbors = rust_lib::read_bin(nsg_trace_path);
    use rayon::prelude::*;
    (start..end).into_par_iter().for_each(move |i| {
        info!("Analyzing sequence {}", i);
        let sequential_result = rust_lib::analyze_sequential(&neighbors, i);
        info!("Analyzing bfs");
        let bfs_result = rust_lib::analyze_knn_bfs(&neighbors, &knn_graph.final_graph_, i);
        info!("Analyzing dfs");
        let dfs_result = rust_lib::analyze_knn_dfs(&neighbors, &knn_graph.final_graph_, i);
        let sequential_file = format!("./s{i}-seq.json");
        serde_json::to_writer(File::create(sequential_file).unwrap(), &sequential_result).unwrap();
        let bfs_file = format!("./s{i}-bfs.json");
        serde_json::to_writer(File::create(bfs_file).unwrap(), &bfs_result).unwrap();
        let dfs_file = format!("./s{i}-dfs.json");
        serde_json::to_writer(File::create(dfs_file).unwrap(), &dfs_result).unwrap();
    });
}
fn translate() {
    let test_path = Path::new("./test1");
    let result = rust_lib::read_neigbors(1000000, test_path);
    rust_lib::translate_neighbors(&result);
}
