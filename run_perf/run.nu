# sudo perf record -o /mnt/raiddisk/sjq/sift-seq.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/sift/sift_200nn.graph ../dataset/sift/sift_base.fvecs seq
sudo perf record -o /mnt/raiddisk/sjq/sift-bfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/sift/sift_200nn.graph ../dataset/sift/sift_base.fvecs bfs
sudo perf record -o /mnt/raiddisk/sjq/sift-dfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/sift/sift_200nn.graph ../dataset/sift/sift_base.fvecs dfs
sudo perf record -o /mnt/raiddisk/sjq/sift-bdfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/sift/sift_200nn.graph ../dataset/sift/sift_base.fvecs bdfs

sudo perf record -o /mnt/raiddisk/sjq/gist-seq.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/gist/gist_100nn.graph ../dataset/gist/gist_base.fvecs seq
sudo perf record -o /mnt/raiddisk/sjq/gist-bfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/gist/gist_100nn.graph ../dataset/gist/gist_base.fvecs bfs
sudo perf record -o /mnt/raiddisk/sjq/gist-dfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/gist/gist_100nn.graph ../dataset/gist/gist_base.fvecs dfs
sudo perf record -o /mnt/raiddisk/sjq/gist-bdfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/gist/gist_100nn.graph ../dataset/gist/gist_base.fvecs bdfs

sudo perf record -o /mnt/raiddisk/sjq/crawl-seq.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/crawl/crawl_100nn.graph ../dataset/crawl/crawl_base.fvecs seq
sudo perf record -o /mnt/raiddisk/sjq/crawl-bfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/crawl/crawl_100nn.graph ../dataset/crawl/crawl_base.fvecs bfs
sudo perf record -o /mnt/raiddisk/sjq/crawl-dfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/crawl/crawl_100nn.graph ../dataset/crawl/crawl_base.fvecs dfs
sudo perf record -o /mnt/raiddisk/sjq/crawl-bdfs.out -e cache-misses,cache-references ~/.cargo/bin/rust-tools build-index-async-with 50 40 500 ../dataset/crawl/crawl_100nn.graph ../dataset/crawl/crawl_base.fvecs bdfs
