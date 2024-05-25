#ifndef EFANNA2E_DFS_H
#define EFANNA2E_DFS_H

#include <memory>
#include <queue>
#include <stack>
#include <string.h>
#include <string>
#include <vector>
struct dfs {

  unsigned current_nodes = 0;
  unsigned total_nodes;
  unsigned knn;
  std::vector<bool> visited;
  std::stack<unsigned> working_stack;

  inline dfs(const unsigned total_nodes, const unsigned knn)
      : total_nodes(total_nodes), knn(knn), visited(total_nodes, false) {}
  inline unsigned next(const unsigned size,
                  const std::vector<std::vector<unsigned>> &data, unsigned *buffer);
};

/// @brief  bfs next
/// @param size the number of nodes to generate this call
/// @param data the graph data n * knn
/// @param buffer the return buffer: size= size
/// @return the real number of nodes generated
inline unsigned dfs::next(const unsigned size,
                     const std::vector<std::vector<unsigned>> &data,
                     unsigned *buffer) {
  // if all nodes are generated, return 0
  if (current_nodes >= total_nodes && working_stack.empty()) {
    return 0;
  }

  unsigned total_generated = 0;
  while (total_generated < size) {
    // if the working queue is empty, generate a new node
    if (working_stack.empty()) {
      // if all nodes are generated, return
      if (current_nodes >= total_nodes) {
        break;
      }
      // generate a new node

      // find the first unvisited node
      while (visited[current_nodes] && current_nodes < total_nodes) {
        current_nodes++;
      }
      // if all nodes are visited, return
      if (current_nodes >= total_nodes) {
        break;
      }
      working_stack.push(current_nodes);
      visited[current_nodes] = true;
      current_nodes++;
    }
    // get the current node
    auto current_node = working_stack.top();
    working_stack.pop();
    buffer[total_generated++] = current_node;
    // get the neighbors of the current node
    for (unsigned i = 0; i < knn; i++) {
      auto neighbor = data[current_node][i];
      // if the neighbor is not visited, add it to the working queue
      if (!visited[neighbor]) {
        working_stack.push(neighbor);
        visited[neighbor] = true;
      }
    }
  }
  return total_generated;
}

#endif // EFANNA2E_DFS_H