#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <efanna2e/index_nsg_sjq.h>
#include <iostream>
template <typename T>
void test_bfs(T &&g, const std::vector<std::vector<unsigned>> &graph,
              unsigned num) {
  // std::vector<std::vector<unsigned>> graph;
  // unsigned dim, num;
  // sjq::load_nn_graph(file_name, graph, dim, num);
  // auto g = bfs(num, dim);
  auto buffer = new unsigned[100];
  auto count = g.next(100, graph, buffer);
  std::vector<bool> visited(num, false);
  while (count != 0) {
    for (unsigned i = 0; i < count; i++) {
      auto node = buffer[i];
      if (visited[node])
        std::cout << "Node " << node << " is visited twice" << std::endl;
      visited[node] = true;
    }
    count = g.next(100, graph, buffer);
  }
  for (unsigned i = 0; i < num; i++) {
    if (!visited[i])
      std::cout << "Node " << i << " is not visited" << std::endl;
  }
}

int main(int argc, char **argv) {
  const char *file_name = argv[1];
  std::vector<std::vector<unsigned>> graph;
  unsigned dim, num;
  sjq::load_nn_graph(file_name, graph, dim, num);
  auto _bfs = bfs(num, dim);
  test_bfs(_bfs, graph, num);
  auto _dfs = dfs(num, dim);
  test_bfs(_dfs, graph, num);
  auto _bdfs = bdfs(num, dim, 10, 50);
  test_bfs(_bdfs, graph, num);
}