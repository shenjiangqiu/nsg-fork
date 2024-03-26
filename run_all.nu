# ./build/tests/test_nsg_index sift_base.fvecs sift_200nn.graph 40 50 500 sift.nsg | save -f nsg_index_sift_200.txt
# ./build/tests/test_nsg_index gist_base.fvecs gist_400nn.graph 60 70 500 gist.nsg | save -f nsg_index_gist_400.txt

~/.cargo/bin/rust-tools build-index-async 50 40 500 ./gist.100nn.graph ./gist_base.fvecs ./result-gist100-async.json
# ~/.cargo/bin/rust-tools build-index-sync 70 60 500 ./gist_400nn.graph ./gist_base.fvecs ./result-gist-sync.json | save -f gist_index_400_sync.txt

~/.cargo/bin/rust-tools build-index-async 50 40 500 ./sift_200nn.graph ./sift_base.fvecs ./result-sift200-async.json
# ~/.cargo/bin/rust-tools build-index-sync 50 40 500 ./sift_200nn.graph ./sift_base.fvecs ./result-sift-sync.json | save -f sift_index_200_sync.txt

~/.cargo/bin/rust-tools build-index-async 50 40 500 ./glove_100nn.graph glove-100/glove-100_base.fvecs ./result-glove100-async.json

