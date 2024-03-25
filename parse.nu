use std log info
let results = ["result-gist-async.json" "result-gist-sync.json" "result-sift-async.json" "result-sift-sync.json"]
for result in $results {
    print $result

    let result_gist_async = open $result
    for r in $result_gist_async {
        print  $"($r.time) ($r.traversal) ($r.num_thread)"
    }
}
