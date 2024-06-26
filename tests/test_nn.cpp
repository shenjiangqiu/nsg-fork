#include <efanna2e/index_nsg_sjq.h>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " data_file" << std::endl;
    exit(-1);
  }

  std::vector<std::vector<unsigned>> graph;
  unsigned dim, num;
  sjq::load_nn_graph(argv[1], graph, dim, num);

  std::cout << "dimension: " << dim << "\n";
  std::cout << "number of nodes: " << num << "\n";
  assert(graph.size() == num);

  std::vector<std::pair<unsigned, unsigned>> counts;
  for (unsigned row_id = 0; row_id < num; row_id++) {
    auto &row = graph[row_id];
    unsigned incorrect =0 ;
    for (unsigned i = 0; i < dim; i++) {
      auto &item = row[i];
      if (item >= num) {
        std::cout << "Invalid node id: " << row_id << " " << i << " " << item
                  << std::endl;
        incorrect++;
      }
    }
    if(incorrect > 0) {
      counts.push_back({row_id, incorrect});
    }
  }
  std::cout << "Graph is valid" << std::endl;
  for (auto &p : counts) {
    std::cout << "Node " << p.first << " has " << p.second << " invalid edges"
              << std::endl;
  }
}