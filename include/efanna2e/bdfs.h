#ifndef EFANNA2E_BDFS_H
#define EFANNA2E_BDFS_H

#include <memory>
#include <queue>
#include <stack>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>
struct bdfs {

  unsigned current_nodes = 0;
  const unsigned total_nodes;
  const unsigned knn;
  const unsigned max_depth;
  const unsigned max_candiate_next;
  std::vector<bool> visited;

  /// @brief the working stack (node, depth)
  std::stack<std::tuple<unsigned, unsigned>> working_stack;

  /// @brief stores the candidate next when depth >= max_depth
  std::vector<int> candidate_next;

  /// @brief constructor
  /// @param total_nodes the number of nodes
  /// @param knn the number of neighbors
  /// @param max_depth the max depth
  inline bdfs(const unsigned total_nodes, const unsigned knn,
              const unsigned max_depth, const unsigned max_candiate_next)
      : total_nodes(total_nodes), knn(knn), max_depth(max_depth),
        max_candiate_next(max_candiate_next), visited(total_nodes, false) {}

  inline unsigned next(const unsigned size,
                       const std::vector<std::vector<unsigned>> &data,
                       unsigned *buffer);
};

/// @brief  bdfs next
/// @param size the number of nodes to generate this call
/// @param data the graph data n * knn
/// @param buffer the return buffer: size= size
/// @return the real number of nodes generated
inline unsigned bdfs::next(const unsigned size,
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
      // first try to find one from candidate next
      int selected = -1;
      for (const auto c : candidate_next) {
        if (!visited[c]) {
          selected = c;
          candidate_next.clear();
          break;
        }
      }

      // if not found, generate a new node
      if (selected == -1) {
        // if all nodes are generated, return
        if (current_nodes >= total_nodes) {
          break;
        }
        // find the first unvisited node
        while (visited[current_nodes] && current_nodes < total_nodes) {
          current_nodes++;
        }
        // if all nodes are visited, return
        if (current_nodes >= total_nodes) {
          break;
        }
        selected = current_nodes;
        current_nodes++;
      }

      if (selected == -1) {
        break;
      }

      // generate a new node
      working_stack.push({selected, 0});
      // the size of depth 0;
      visited[selected] = true;
    }

    const auto [current_node, current_depth] = working_stack.top();
    // get the current node
    working_stack.pop();

    buffer[total_generated++] = current_node;

    if (current_depth >= max_depth) {
      // add some to candidate next
      unsigned i = 0;
      while (i < knn && candidate_next.size() < (unsigned)max_candiate_next) {
        int neighbor = data[current_node][i];
        if (!visited[neighbor]) {
          candidate_next.push_back(neighbor);
        }
        i++;
      }
    } else {
      // add to next depth
      // get the neighbors of the current node
      for (unsigned i = 0; i < knn; i++) {
        int neighbor = data[current_node][i];
        // if the neighbor is not visited, add it to the working queue
        if (!visited[neighbor]) {
          working_stack.push({neighbor, current_depth + 1});
          visited[neighbor] = true;
        }
      }
    }
  }
  return total_generated;
}

#endif // EFANNA2E_BDFS_H