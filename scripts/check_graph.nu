let graphs = ["raidsjq/spacev.100M.100NN.graph"]
for g in $graphs {
    use std log info
    info $"Running ($g)"
    rust-tools check-knn-graph $g
}