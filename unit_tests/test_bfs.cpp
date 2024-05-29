#include <catch2/catch_test_macros.hpp>
#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <efanna2e/index_nsg_sjq.h>
#include <iostream>
#include <vector>
TEST_CASE("bfs_correct") {
  std::vector<std::vector<unsigned>> data;
  unsigned dim, num;
  sjq::load_nn_graph(
      "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/sift_100M/out_10M.ivecs",
      data, dim, num);
  bfs bfs(num, dim);
  std::vector<bool> visited(num, false);
  unsigned buffer[50 * 100];
  unsigned count = bfs.next(50 * 100, data, buffer);
  while (count != 0) {
    REQUIRE(count <= 50 * 100);
    for (unsigned i = 0; i < count; i++) {
      REQUIRE(buffer[i] < num);
      if (visited[buffer[i]]) {
        std::cout << "error" << std::endl;
        std::cout << "buffer[i]:" << buffer[i] << " is already visited"
                  << std::endl;
      }
      visited[buffer[i]] = true;
    }

    count = bfs.next(50 * 100, data, buffer);
  }
  for (unsigned i = 0; i < num; i++) {
    if (!visited[i]) {
      std::cout << "error" << std::endl;
      std::cout << "i:" << i << " is not visited" << std::endl;
    }
  }
}

TEST_CASE("bfs_correct_spaceev") {
  std::vector<std::vector<unsigned>> data;
  unsigned dim, num;
  sjq::load_nn_graph(
      "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/spaceev/out_10M.ivecs",
      data, dim, num);
  bfs bfs(num, dim);
  std::vector<bool> visited(num, false);
  unsigned buffer[50 * 100];
  unsigned count = bfs.next(50 * 100, data, buffer);
  while (count != 0) {
    REQUIRE(count <= 50 * 100);
    for (unsigned i = 0; i < count; i++) {
      REQUIRE(buffer[i] < num);
      if (visited[buffer[i]]) {
        std::cout << "error" << std::endl;
        std::cout << "buffer[i]:" << buffer[i] << " is already visited"
                  << std::endl;
      }
      visited[buffer[i]] = true;
    }

    count = bfs.next(50 * 100, data, buffer);
  }
  for (unsigned i = 0; i < num; i++) {
    if (!visited[i]) {
      std::cout << "error" << std::endl;
      std::cout << "i:" << i << " is not visited" << std::endl;
    }
  }
}

TEST_CASE("bfs_correct_deep") {
  std::vector<std::vector<unsigned>> data;
  unsigned dim, num;
  sjq::load_nn_graph(
      "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/deep/out_10M.ivecs",
      data, dim, num);
  bfs bfs(num, dim);
  std::vector<bool> visited(num, false);
  unsigned buffer[50 * 100];
  unsigned count = bfs.next(50 * 100, data, buffer);
  while (count != 0) {
    REQUIRE(count <= 50 * 100);
    for (unsigned i = 0; i < count; i++) {
      REQUIRE(buffer[i] < num);
      if (visited[buffer[i]]) {
        std::cout << "error" << std::endl;
        std::cout << "buffer[i]:" << buffer[i] << " is already visited"
                  << std::endl;
      }
      visited[buffer[i]] = true;
    }

    count = bfs.next(50 * 100, data, buffer);
  }
  for (unsigned i = 0; i < num; i++) {
    if (!visited[i]) {
      std::cout << "error" << std::endl;
      std::cout << "i:" << i << " is not visited" << std::endl;
    }
  }
}