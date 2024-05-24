#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
class TaskQueue {
public:
  void push(std::function<void()> task) { tasks.push(task); }

  std::function<void()> pop() {
    if (tasks.empty()) {
      return nullptr;
    }
    auto task = tasks.front();
    tasks.pop();
    return task;
  }

  bool empty() { return tasks.empty(); }

private:
  std::queue<std::function<void()>> tasks;
};

void taskProducer(TaskQueue &taskQueue, int n, std::mutex &cvMtx,
                  std::condition_variable &cv, bool &ready) {
  while (true) {
    std::unique_lock<std::mutex> lock(cvMtx);
    cv.wait(lock, [&ready] { return !ready; });
    std::cout << "Producer is producing tasks" << std::endl;
    for (int i = 0; i < n; ++i) {
      taskQueue.push([i]() {
        std::cout << "Task " << i << " is being processed by thread "
                  << std::this_thread::get_id() << std::endl;
      });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Producer has finished" << std::endl;

    ready = true;
    cv.notify_all();
  }
}

void taskConsumer(TaskQueue &taskQueue, std::mutex &cvMtx,
                  std::condition_variable &cv, bool &ready,
                  unsigned numConsumers, std::atomic_uint &tasksCompleted) {
  while (true) {
    std::cout << "Thread " << std::this_thread::get_id()
              << " is waiting for tasks\n";
    std::unique_lock<std::mutex> lock(cvMtx);
    cv.wait(lock, [&ready] { return ready; });
    auto task = taskQueue.pop();
    lock.unlock();

    if (task) {
      std::cout << "Thread " << std::this_thread::get_id()
                << " is processing a task" << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      task();
      std::cout << "Thread " << std::this_thread::get_id() << " has finished"
                << std::endl;
    }

    ++tasksCompleted;
    if (tasksCompleted == numConsumers) {
      ready = false;
      tasksCompleted = 0;
      cv.notify_all();
    }
  }
}

int main() {
  const unsigned numThreads = 5;
  TaskQueue taskQueue;
  std::mutex cvMtx;
  std::condition_variable cv;
  bool ready = false;
  std::atomic_uint tasksCompleted(0);

  std::thread producerThread(taskProducer, std::ref(taskQueue), numThreads,
                             std::ref(cvMtx), std::ref(cv), std::ref(ready));

  std::vector<std::thread> consumerThreads;
  for (unsigned i = 0; i < numThreads; ++i) {
    consumerThreads.emplace_back(taskConsumer, std::ref(taskQueue),
                                 std::ref(cvMtx), std::ref(cv), std::ref(ready),
                                 numThreads, std::ref(tasksCompleted));
  }

  producerThread.join();
  for (auto &thread : consumerThreads) {
    thread.join();
  }

  return 0;
}
