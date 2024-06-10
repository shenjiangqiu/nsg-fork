#ifndef ASYNC_TASK_POOL_H
#define ASYNC_TASK_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
constexpr unsigned num_partitions = 4;

class AsyncTaskPool {
public:
  inline AsyncTaskPool(unsigned num_threads, unsigned num_tasks)
      : buffer(new unsigned[num_threads * (num_tasks + 1) * num_partitions]),
        num_threads(num_threads), num_tasks(num_tasks) {}
  inline ~AsyncTaskPool() { delete[] buffer; }

  /// # getting a task
  /// - all consumers can call this function
  inline bool get_next_task(std::function<bool(unsigned *)> consumer) {
    unsigned current_partition;
    unsigned current_task;
    unsigned *task;
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [this] { return this->reading_partitions > 0; });
      current_partition = this->current_reading_partition;
      current_task = this->current_read_task;
      // std::cout << "current_partition:" << current_partition << std::endl;
      // std::cout << "current_task:" << current_task << std::endl;
      this->current_read_task++;
      task = buffer + current_partition * num_threads * (num_tasks + 1) +
             current_task * (num_tasks + 1);

      if (this->current_read_task == num_threads) {
        // need to move to next partition
        this->reading_partitions--;
        this->current_reading_partition++;
        this->current_reading_partition %= num_partitions;
        this->current_read_task = 0;
      }
    }
    bool good = consumer(task);
    {
      //
      std::unique_lock<std::mutex> lock(mutex);
      this->partition_remaining_working[current_partition]--;
      if (current_partition == this->current_valid_partition &&
          this->partition_remaining_working[current_partition] == 0) {
        this->current_valid_partition++;
        this->current_valid_partition %= num_partitions;
        this->empty_partitions++;
        // recursive test the next partition
        unsigned all_valid_partitions = num_partitions - this->empty_partitions;
        while (all_valid_partitions > 0) {
          if (this->partition_remaining_working
                  [this->current_valid_partition] == 0) {
            this->current_valid_partition++;
            this->current_valid_partition %= num_partitions;
            this->empty_partitions++;
            all_valid_partitions--;
          } else {
            break;
          }
        }
        cv.notify_all();
      }
    }

    return good;

    // consume the task
  }

  /// # generating a task
  /// - make sure only one thread can call this function
  /// - when a task is generated, all threads waiting for get_next_task() can
  /// get the task
  inline bool set_next_task(std::function<bool(unsigned *)> generator) {
    // first get the buffer to store the tasks
    unsigned next_partition;
    unsigned *task_buffer;
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [this] { return this->empty_partitions > 0; });
      next_partition = this->next_write_partition;
      this->next_write_partition++;
      this->next_write_partition %= num_partitions;
      this->empty_partitions--;
      this->partition_remaining_working[next_partition] = num_threads;
      // valid_partitions++;
      task_buffer = buffer + next_partition * num_threads * (num_tasks + 1);
    }

    bool good = generator(task_buffer);

    // generate the tasks

    {
      std::unique_lock<std::mutex> lock(mutex);
      this->reading_partitions++;
      cv.notify_all();
    }
    return good;
  }

private:
  // init by initialize

  /// @brief  the buffer store the tasks, the size is num_partitions *
  /// num_threads * num_tasks
  unsigned *buffer;
  unsigned num_threads;
  unsigned num_tasks;

  // init default

  // --- for re-use
  /// @brief  the remaining working tasks in each partition, when finsihed task
  /// of that partition, decrease the value, when it reach zero, the partition
  /// is ready to write.
  unsigned partition_remaining_working[num_partitions] = {0};

  /// @brief  the current partition that is still valid( might be reading, don't
  /// write now)
  unsigned current_valid_partition = 0;

  // --- for reading

  /// @brief  the number of partitions that are valid to be read(added by
  /// generator)
  unsigned reading_partitions = 0;

  /// @brief  the current partition that is valid to be read(added by generator)
  unsigned current_reading_partition = 0;

  /// @brief  the id of the task inside a partition that should be read next
  unsigned current_read_task = 0;

  // --- for writing

  /// @brief  the id of the partition that should be written next
  unsigned next_write_partition = 0;

  unsigned empty_partitions = num_partitions;

  std::mutex mutex;
  std::condition_variable cv;
};

#endif // ASYNC_TASK_POOL_H