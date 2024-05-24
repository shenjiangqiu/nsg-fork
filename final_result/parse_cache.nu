let files = (ls *cache.json)
for file in $files {
    print $file.name
    let records = open $file.name
    for record in $records {
        let results = $record.result
        let name = $record.traversal
        print --no-newline $"name: ($name) "
        for r in $results {
            print --no-newline $"($r.total) ($r.misses) "
        }
        print ""
    }
}
