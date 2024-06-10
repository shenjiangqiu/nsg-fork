#include <catch2/catch_test_macros.hpp>
#include <efanna2e/async_task_pool.h>
#include <efanna2e/bfs.h>
#include <efanna2e/index_nsg_sjq.h>
#include <iostream>
#include <thread>
#include <vector>

TEST_CASE("test_task") {
  std::vector<std::vector<unsigned>> data;
  unsigned dim, num;
  sjq::load_nn_graph(
      "./dataset/sift/sift_200nn.graph",
      // "/mnt/raiddisk/sjq/generate_faiss_knn/dataset/sift_100M/out_10M.ivecs",
      data, dim, num);

  bfs bfs(num, dim);
  const unsigned threads = 10;
  const unsigned task_size = 100;
  AsyncTaskPool task_pool(threads, task_size);

  std::thread producer([&] {
    std::cout << "producer" << std::endl;
    unsigned *temp_buffer = new unsigned[threads * task_size];
    while (true) {
      bool good = task_pool.set_next_task([&](unsigned *buffer) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        unsigned size = (threads * task_size);
        auto total_tasks = bfs.next(size, data, temp_buffer);

        for (unsigned i = 0; i < threads; i++) {
          buffer[i * (task_size + 1)] = 0;
        }
        // put temp_buffer to buffer
        for (unsigned i = 0; i < total_tasks; i++) {
          unsigned index = i % threads;
          unsigned current_task_index = buffer[index * (task_size + 1)];
          buffer[index * (task_size + 1) + 1 + current_task_index] =
              temp_buffer[i];
          buffer[index * (task_size + 1)]++;
        }
        if (total_tasks == 0) {
          return false;
        } else {
          return true;
        }
      });
      if (!good) {
        break;
      }
    }
    delete[] temp_buffer;
  });
  std::vector<std::thread> consumers;
  for (int i = 0; i < 10; i++) {
    consumers.push_back(std::thread([&] {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto good = task_pool.get_next_task([](unsigned *task) {
          unsigned tasks = task[0];
          assert(tasks <= task_size);
          if (tasks == 0) {
            return false;
          }
          for (unsigned i = 0; i < tasks; i++) {
            std::cout << task[i + 1] << " ";
          }
          // std::cout << std::endl;
          return true;
        });
        if (!good) {
          break;
        }
      }
    }));
  }

  producer.join();
  for (auto &consumer : consumers) {
    consumer.join();
  }
}