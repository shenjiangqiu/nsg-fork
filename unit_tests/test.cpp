#include <catch2/catch_test_macros.hpp>
#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <iostream>

TEST_CASE("bfs next", "[bfs]") {
  std::cout << "bfs next" << std::endl;
  int data[10] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2};
  int buffer[2];
  bfs bfs(5, 2);
  int size = 2;
  int ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 2);
  REQUIRE(buffer[0] == 0);
  REQUIRE(buffer[1] == 1);

  ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 2);
  REQUIRE(buffer[0] == 2);
  REQUIRE(buffer[1] == 3);

  ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 1);
  REQUIRE(buffer[0] == 4);

  ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 0);
}

TEST_CASE("bfs-4", "[bfs]") {
  std::cout << "bfs 4" << std::endl;
  int data[10] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2};
  int buffer[4];
  bfs bfs(5, 2);
  int size = 4;
  int ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 4);
  REQUIRE(buffer[0] == 0);
  REQUIRE(buffer[1] == 1);
  REQUIRE(buffer[2] == 2);
  REQUIRE(buffer[3] == 3);

  ret = bfs.next(size, data, buffer);

  REQUIRE(ret == 1);
  REQUIRE(buffer[0] == 4);

  ret = bfs.next(size, data, buffer);
  REQUIRE(ret == 0);
}

TEST_CASE("dfs next", "[dfs]") {
  std::cout << "dfs next" << std::endl;
  int data[10] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2};
  int buffer[2];
  dfs dfs(5, 2);
  int size = 2;
  int ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 2);
  REQUIRE(buffer[0] == 0);
  REQUIRE(buffer[1] == 2);

  ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 2);
  REQUIRE(buffer[0] == 1);
  REQUIRE(buffer[1] == 4);

  ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 1);
  REQUIRE(buffer[0] == 3);

  ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 0);
}

TEST_CASE("dfs-4", "[dfs]") {
  std::cout << "dfs 4" << std::endl;
  int data[10] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2};
  int buffer[4];
  dfs dfs(5, 2);
  int size = 4;
  int ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 4);
  REQUIRE(buffer[0] == 0);
  REQUIRE(buffer[1] == 2);
  REQUIRE(buffer[2] == 1);
  REQUIRE(buffer[3] == 4);

  ret = dfs.next(size, data, buffer);

  REQUIRE(ret == 1);
  REQUIRE(buffer[0] == 3);

  ret = dfs.next(size, data, buffer);
  REQUIRE(ret == 0);
}

TEST_CASE("bdfs-4", "[bdfs]") {
  std::cout << "bdfs 4" << std::endl;
  int data[10] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2};
  int buffer[4];
  bdfs bdfs(5, 2, 2, 2);
  int size = 4;
  int ret = bdfs.next(size, data, buffer);
  REQUIRE(ret == 4);
  REQUIRE(buffer[0] == 0);
  REQUIRE(buffer[1] == 2);
  REQUIRE(buffer[2] == 1);
  REQUIRE(buffer[3] == 4);

  ret = bdfs.next(size, data, buffer);

  REQUIRE(ret == 1);
  REQUIRE(buffer[0] == 3);

  ret = bdfs.next(size, data, buffer);
  REQUIRE(ret == 0);
}
