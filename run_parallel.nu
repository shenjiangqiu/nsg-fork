# ./build_release/tests/sjq_test_nsg_index dataset/sift/sift_base.fvecs dataset/sift/sift_200nn.graph 50 40 500 out.nsg 40 100
# ./build_release/tests/sjq_test_nsg_index dataset/sift/sift_base.fvecs dataset/sift/sift_200nn.graph 50 40 500 out.nsg 40 100

let dataset_base_dir = "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/"
let dataset = ["sift_100M","spaceev","deep"]
# let dataset = ["spaceev"]
let base_100M = "base_100M.fvecs"
let out_100M = "out_100M.ivecs"
let base_10M = "base_10M.fvecs"
let out_10M = "out_10M.ivecs"
use std log info
# for config in [ [$base_10M,$out_10M],[$base_100M,$out_100M]] {
for config in [ [$base_10M,$out_10M]] {
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
    ./build_release/tests/test_nsg_index $fvecs_file $ivecs_file 50 40 500 out.nsg 80 |save -f $"out_nsg_old_($data)_($base_file).log" 
    # ./build_release/tests/test_nsg_index $fvecs_file $ivecs_file 50 40 500 out.nsg 41  

    ./build_release/tests/sjq_test_nsg_index $fvecs_file $ivecs_file 50 40 500 out.nsg 79 100 |save -f $"out_nsg_new_($data)_($base_file).log"
  }
}
