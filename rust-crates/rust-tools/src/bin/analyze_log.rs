use std::path::PathBuf;
use clap::Parser;
use regex::Regex;
#[derive(Parser)]
struct Cli {
    log_path: PathBuf,
}
fn main() {
  let cli = Cli::parse();
}
