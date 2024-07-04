# ./build_release/tests/sjq_test_nsg_index dataset/sift/sift_base.fvecs dataset/sift/sift_200nn.graph 50 40 500 out.nsg 40 100
# ./build_release/tests/sjq_test_nsg_index dataset/sift/sift_base.fvecs dataset/sift/sift_200nn.graph 50 40 500 out.nsg 40 100

let dataset_base_dir = "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/"
let dataset = ["sift_100M","spaceev","deep"]
# let dataset = ["spaceev"]
let base_100M = "base_100M.fvecs"
let out_100M = "out_100M.ivecs"
let base_10M = "base_10M.fvecs"
let out_10M = "out_10M.ivecs"
let base_1M = "base_1M.fvecs"
let out_1M = "out_1M.ivecs"
let date = date now|format date "%m_%d_%H_%M_%S"
let sync_point = [50 100 200 400 800 2000]
let current_config = [ [$base_10M,$out_10M]]
mkdir  $"logs/($date)"
# for thread in [30 40 50 60 70 ] {
  for thread in [ 40  ] {
  
  for batch in [1 ] {
    use std log info
    info $"runing for threads: ($thread)"
    # for config in [ [$base_10M,$out_10M],[$base_100M,$out_100M]] {
    for config in $current_config { 
      let base_file = $config|get 0
      let out_file = $config|get 1
      for data in $dataset {
        info $data
        # run 10M
        let fvecs_file = ($dataset_base_dir|path join $data|path join $base_file )
        info $fvecs_file
        let ivecs_file = ($dataset_base_dir|path join $data|path join $out_file )
        info $ivecs_file
        just build_release
        # ./build_release/tests/test_nsg_index $fvecs_file $ivecs_file 50 40 500 out.nsg $thread $batch |save -f $"logs/($date)/out_nsg_old_($data)_($base_file)_($thread)_($batch).log" 
        # ./build_release/tests/test_nsg_index $fvecs_file $ivecs_file 50 40 500 out.nsg $thread $batch 
        ./build_release/tests/sjq_test_nsg_index_static_omp_static $fvecs_file $ivecs_file 50 40 500 out.nsg $thread $batch |save -f $"logs/($date)/out_nsg_new_static_omp_static_($data)_($base_file)_($thread)_($batch).log";
        
        for sync in $sync_point {
          ./build_release/tests/sjq_test_nsg_index_static_omp_dynamic $fvecs_file $ivecs_file 50 40 500 out.nsg $thread $batch $sync |save -f $"logs/($date)/out_nsg_new_static_omp_dynamic_($data)_($base_file)_($thread)_($batch)_($sync).log";
        }
        
      }
    } 
  }
  
  
}
