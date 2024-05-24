#ifndef EFANNA2E_DFS_H
#define EFANNA2E_DFS_H

#include <memory>
#include <queue>
#include <stack>
#include <string.h>
#include <string>
#include <vector>
struct dfs {

  int current_nodes = 0;
  int total_nodes;
  int knn;
  std::vector<bool> visited;
  std::stack<int> working_stack;

  inline dfs(const int total_nodes, const int knn)
      : total_nodes(total_nodes), knn(knn), visited(total_nodes, false) {}
  inline int next(const int size, const int *data, int *buffer);
};

/// @brief  bfs next
/// @param size the number of nodes to generate this call
/// @param data the graph data n * knn
/// @param buffer the return buffer: size= size
/// @return the real number of nodes generated
inline int dfs::next(const int size, const int *data, int *buffer) {
  // if all nodes are generated, return 0
  if (current_nodes >= total_nodes && working_stack.empty()) {
    return 0;
  }

  int total_generated = 0;
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
    int current_node = working_stack.top();
    working_stack.pop();
    buffer[total_generated++] = current_node;
    // get the neighbors of the current node
    for (int i = 0; i < knn; i++) {
      int neighbor = data[current_node * knn + i];
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