
~/.cargo/bin/rust-tools build-index-async 50 40 500 ./dataset/crawl/crawl_100nn.graph ./dataset/crawl/crawl_base.fvecs ./final_result/result_craws.json
~/.cargo/bin/rust-tools build-index-async 50 40 500 ./dataset/glove/glove_100nn.graph ./dataset/glove/glove-100_base.fvecs ./final_result/result_glove.json
~/.cargo/bin/rust-tools build-index-async 50 40 500 ./dataset/gist/gist_100nn.graph ./dataset/gist/gist_base.fvecs ./final_result/result_gist.json
~/.cargo/bin/rust-tools build-index-async 50 40 500 ./dataset/sift/sift_200nn.graph ./dataset/sift/sift_base.fvecs ./final_result/result_sift.json

~/.cargo/bin/rust-tools build-index-async-limited 50 40 500 /mnt/raiddisk/sjq/deep.100M.100NN.graph /mnt/raiddisk/sjq/deep_100M_base.fvecs ./final_result/result_depp100m.json
~/.cargo/bin/rust-tools build-index-async-limited 50 40 500 /mnt/raiddisk/sjq/sift100M.100NN.graph /mnt/raiddisk/sjq/sift_100M.fvecs ./final_result/result_sift100m.json
~/.cargo/bin/rust-tools build-index-async-limited 50 40 500 /home/sjq/git/nsg-fork/raidsjq/spacev.100M.100NN.graph /home/sjq/git/nsg-fork/raidsjq/spacev100m_base.fvecs ./final_result/result_spacev100m.json