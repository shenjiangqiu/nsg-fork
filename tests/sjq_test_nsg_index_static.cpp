//
// Created by 付聪 on 2017/6/21.
//

// #include <efanna2e/index_nsg_sjq.h>
#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <efanna2e/index_nsg_sjq_static.h>
#include <efanna2e/util.h>

#include <omp.h>
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
    std::cout << argv[0] << " data_file nn_graph_path L R C save_graph_file T B"
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
  unsigned B = (unsigned)atoi(argv[8]);
  // unsigned batchSize = (unsigned)atoi(argv[8]);
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<unsigned>("T", T);
  paras.Set<unsigned>("B", B);
  omp_set_num_threads(T);
  std::cerr << "threads: " << T << "\n";
  // paras.Set<unsigned>("batchSize", batchSize);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  sjq_static::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

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
    for (unsigned round = 0; round < 5; round++) {
      srand(22233);
      std::cout << "traversal_idx: " << traversal_idx << "\n";
      std::cout << "round: " << round << "\n";
      std::cout.flush();
      // const unsigned long *trace = traversal_sequence[0];
      auto start_traversal = std::chrono::high_resolution_clock::now();
      unsigned *traversal_sequence = new unsigned[index.nd_];
      if (traversal_idx == 1) {
        // it's bfs
        bfs m_bfs(index.nd_, index.knn);
        m_bfs.next(index.nd_, index.final_graph_, traversal_sequence);

      } else if (traversal_idx == 2) {
        // it's dfs
        dfs m_dfs(index.nd_, index.knn);
        m_dfs.next(index.nd_, index.final_graph_, traversal_sequence);
      } else if (traversal_idx == 3) {
        // it's bdfs
        bdfs m_bdfs(index.nd_, index.knn, 25, 100);
        m_bdfs.next(index.nd_, index.final_graph_, traversal_sequence);
      } else {
        // it's error
        throw std::runtime_error("traversal_idx is error");
      }
      // paras.Set<unsigned>("Threads", threads);

      auto end_traversal = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff_traversal =
          end_traversal - start_traversal;
      std::cout << "traversal time: " << diff_traversal.count() << "\n";

      // validate the traversal sequence
      std::cout << "validating the traversal sequence\n" << std::endl;
      std::vector<bool> visited(index.nd_, false);
      for (unsigned i = 0; i < index.nd_; i++) {
        if (traversal_sequence[i] >= index.nd_) {
          std::cout << "Invalid node id: " << i << " " << traversal_sequence[i]
                    << std::endl;
        }
        if (visited[traversal_sequence[i]]) {
          std::cout << "Node " << traversal_sequence[i] << " is visited twice"
                    << std::endl;
        }
        visited[traversal_sequence[i]] = true;
      }
      for (unsigned i = 0; i < index.nd_; i++) {
        if (!visited[i]) {
          std::cout << "Node " << i << " is not visited" << std::endl;
        }
      }
      std::cout << "end validating the traversal sequence\n" << std::endl;

      auto start_nsg = std::chrono::high_resolution_clock::now();

      index.Build_static(points_num, data_load, paras, traversal_sequence);
      auto end_nsg = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_nsg = end_nsg - start_nsg;

      std::cout << "indexing time: " << duration_nsg.count() << "\n";
      std::chrono::duration<double> diff_index = duration_nsg + diff_traversal;
      std::cout << "total time: " << diff_index.count() << "\n";
    }
  }
  // }

  // index.Save(argv[6]);

  return 0;
}
