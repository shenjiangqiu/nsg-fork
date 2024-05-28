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

  for (const auto &row : graph) {
    for (const auto &item : row) {
      if (item >= num) {
        std::cout << "Invalid node id: " << item << std::endl;
        exit(-1);
      }
    }
  }
  std::cout << "Graph is valid" << std::endl;
}