#include <catch2/catch_test_macros.hpp>
#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <efanna2e/index_nsg_sjq_static_omp_static.h>
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>

#include <barrier>
TEST_CASE("omp_schedule") {

  omp_set_num_threads(10);
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int i = 0; i < 100; i++) {
      std::cout << "i: " << i << " : " << omp_get_thread_num() << std::endl;
    }
  }
}

TEST_CASE("distribute_sequence") {
  unsigned *traversal_sequence = new unsigned[10];
  for (unsigned i = 0; i < 10; i++) {
    traversal_sequence[i] = i;
  }

  unsigned nd_rounded = (10 + 4 - 1) / 4 * 4;
  REQUIRE(nd_rounded == 12);
  unsigned *traversal_sequence_reordered = new unsigned[nd_rounded];

  sjq_static_omp_static::distribute_traversal_sequence(
      traversal_sequence, traversal_sequence_reordered, 4, 10, nd_rounded);
  for (unsigned i = 0; i < 12; i++) {
    std::cout << traversal_sequence_reordered[i] << " ";
  }
  REQUIRE(traversal_sequence_reordered[8] == (unsigned)-1);
}

TEST_CASE("test_syncpoint") {
  omp_set_num_threads(10);
  std::barrier barrier(10);
#pragma omp parallel
  {
    unsigned count = 0;
#pragma omp for schedule(static)
    for (int i = 0; i < 100; i++) {
      std::cout << "i: " << i << " : " << omp_get_thread_num() << std::endl;
      count++;
      if (count == 5) {
        std::cout << "syncpoint" << std::endl;
        barrier.arrive_and_wait();
      }
    }
  }
}