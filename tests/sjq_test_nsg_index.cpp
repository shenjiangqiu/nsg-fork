//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg_sjq.h>
#include <efanna2e/util.h>
// #include "rust-lib.h"

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim) { // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();
}

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cout << argv[0]
              << " data_file nn_graph_path L R C save_graph_file T batchSize"
              << std::endl;
    exit(-1);
  }
  float *data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);
  unsigned T = (unsigned)atoi(argv[7]);
  unsigned batchSize = (unsigned)atoi(argv[8]);
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<unsigned>("T", T);
  paras.Set<unsigned>("batchSize", batchSize);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  sjq::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

  auto dm = index.GetDimension();
  auto sz = index.GetSizeOfDataset();
  std::cout << "dimension: " << dm << "\n";
  std::cout << "size of dataset: " << sz << "\n";
  std::cout << "loading  traversal data" << "\n";
  // auto traversal_sequence = build_traversal_seqence(nn_graph_path.c_str());
  std::cout << "done, loading nn graph" << "\n";
  index.Load_nn_graph(nn_graph_path.c_str());
  std::cout << "done, start runing" << "\n";
  // for (unsigned threads = 64; threads >= 4; threads -= 4) {
  for (unsigned traversal_idx = 1; traversal_idx <= 3; traversal_idx++) {
    std::cout << "traversal_idx: " << traversal_idx << "\n";
    // const unsigned long *trace = traversal_sequence[0];
    auto s = std::chrono::high_resolution_clock::now();

    // paras.Set<unsigned>("Threads", threads);

    index.Build(points_num, data_load, paras, traversal_idx);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
  }
  // }

  // index.Save(argv[6]);

  return 0;
}
