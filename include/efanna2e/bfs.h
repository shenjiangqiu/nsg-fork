#ifndef EFANNA2E_BFS_H
#define EFANNA2E_BFS_H

#include <memory>
#include <queue>
#include <string.h>
#include <string>
#include <vector>
struct bfs {

  int current_nodes = 0;
  int total_nodes;
  int knn;
  std::vector<bool> visited;
  std::queue<int> working_queue;

  inline bfs(const int total_nodes, const int knn)
      : total_nodes(total_nodes), knn(knn), visited(total_nodes, false) {}
  inline int next(const int size, const int *data, int *buffer);
};


/// @brief  bfs next
/// @param size the number of nodes to generate this call
/// @param data the graph data n * knn
/// @param buffer the return buffer: size= size
/// @return the real number of nodes generated
inline int bfs::next(const int size, const int *data, int *buffer) {
  // if all nodes are generated, return 0
  if (current_nodes >= total_nodes && working_queue.empty()) {
    return 0;
  }

  int total_generated = 0;
  while (total_generated < size) {
    // if the working queue is empty, generate a new node
    if (working_queue.empty()) {
      // if all nodes are generated, return
      if (current_nodes >= total_nodes) {
        break;
      }
      // generate a new node

      // find the first unvisited node
      while(visited[current_nodes] && current_nodes < total_nodes){
        current_nodes++;
      }
      // if all nodes are visited, return
      if (current_nodes >= total_nodes) {
        break;
      }
      working_queue.push(current_nodes);
      visited[current_nodes] = true;
      current_nodes++;
    }
    // get the current node
    int current_node = working_queue.front();
    working_queue.pop();
    buffer[total_generated++] = current_node;
    // get the neighbors of the current node
    for (int i = 0; i < knn; i++) {
      int neighbor = data[current_node * knn + i];
      // if the neighbor is not visited, add it to the working queue
      if (!visited[neighbor]) {
        working_queue.push(neighbor);
        visited[neighbor] = true;
      }
    }
  }
  return total_generated;
}

#endif // EFANNA2E_BFS_H