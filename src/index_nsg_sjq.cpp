#include "efanna2e/index_nsg.h"

#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cmath>
#include <omp.h>
// #include <rust-lib.h>
#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
// #include <efanna2e/avx512.h>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <efanna2e/avx256.h>
#include <efanna2e/bdfs.h>
#include <efanna2e/bfs.h>
#include <efanna2e/dfs.h>
#include <efanna2e/index_nsg_sjq.h>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
namespace sjq {
#define _CONTROL_NUM 100

IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   std::unique_ptr<Index> &&initializer)
    : initializer_(std::move(initializer)), dimension_(dimension), nd_(n),
      has_built(false) {

  switch (m) {
  case L2:
    distance_ = std::unique_ptr<DistanceL2>(new DistanceL2());
    break;
  default:
    distance_ = std::unique_ptr<DistanceL2>(new DistanceL2());
    break;
  }
}

IndexNSG::~IndexNSG() {}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof())
      break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  // std::cout<<cc<<std::endl;
}
void load_nn_graph(const char *filename,
                   std::vector<std::vector<unsigned>> &final_graph_,
                   unsigned &dim, unsigned &num) {
  std::cout << "loading graph file: " << filename << std::endl;
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&dim, sizeof(unsigned));
  std::cout << "k: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  if (num * (dim + 1) * 4 != fsize) {
    std::cout << "file size error" << std::endl;
    exit(-1);
  }
  std::cout << "num: " << num << std::endl;
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (dim + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(dim);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), dim * sizeof(unsigned));
    if (i == 0) {
      for (int j = 0; j < 10; j++) {
        std::cout << "final_graph_[0]: " << final_graph_[0][j] << std::endl;
      }
    }
  }
  in.close();
}
void IndexNSG::Load_nn_graph(const char *filename) {
  std::cout << "loading graph file: " << filename << std::endl;
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  this->knn = k;
  std::cout << "k: " << k << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  if (num != nd_) {
    std::cout << "num: " << num << " nd_: " << nd_ << std::endl;
    throw std::invalid_argument("Invalid graph file");
  }
  if (num * (k + 1) * 4 != fsize) {
    std::cout << "file size error" << std::endl;
    exit(-1);
  }
  std::cout << "num: " << num << std::endl;
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
    if (i == 0) {
      for (int j = 0; j < 10; j++) {
        std::cout << "final_graph_[0]: " << final_graph_[0][j] << std::endl;
      }
    }
  }
  in.close();
  // for(int i=0;i<10;i++){
  //   for(auto& node : final_graph_[i]){
  //     std::cout<<node<<" ";
  //   }
  //   std::cout<<std::endl;
  // }

  // for(int i=num-10;i<num;i++){
  //   for(auto& node : final_graph_[i]){
  //     std::cout<<node<<" ";
  //   }
  //   std::cout<<std::endl;
  // }
  // exit(0);
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size();
       i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id])
      continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_)
      continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id])
          continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance)
          continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size())
          ++L;
        if (r < nk)
          nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  // get L nodes from the neighbor of final_graph_[ep_]
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size();
       i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id])
      continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_)
      continue;
    // std::cout<<id<<std::endl;
    // the distance from query to id
    // float dist = computeL2Distance(data_ + dimension_ * (size_t)id, query,
    //                                 (unsigned)dimension_);
    // float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
    //                                 (unsigned)dimension_);
    // float dist = avx512l2translated(data_ + dimension_ * (size_t)id, query,
    //                                 (unsigned)dimension_);
    float dist = avx256l2translated(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }

  // sort by distance from low to high
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;
    // push the node's neighbor into retset
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id])
          continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance)
          continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size())
          ++L;
        if (r < nk)
          nk = r;
      }
    }
    // after test one node in L, check the next node
    // every time we insert a smaller node than current node, restart from that
    // node
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++)
    center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_; // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
  delete[] center;
}

void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {
  // pool is the full set
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  width = range;
  unsigned start = 0;
  // push the original K-nodes
  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id])
      continue;
    // float dist =
    //     computeL2Distance(data_ + dimension_ * (size_t)q,
    //                        data_ + dimension_ * (size_t)id,
    //                        (unsigned)dimension_);
    // float dist = distance_->compare(data_ + dimension_ * (size_t)q,
    //                                 data_ + dimension_ * (size_t)id,
    //                                 (unsigned)dimension_);
    // float dist = avx512l2translated(data_ + dimension_ * (size_t)q,
    //                                 data_ + dimension_ * (size_t)id,
    //                                 (unsigned)dimension_);
    float dist = avx256l2translated(data_ + dimension_ * (size_t)q,
                                    data_ + dimension_ * (size_t)id,
                                    (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q)
    start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    if (!occlude)
      result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1)
      break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1)
          break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup)
      continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(
              data_ + dimension_ * (size_t)result[t].id,
              data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude)
          result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range)
            des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

__attribute__((unused)) static void save_pool(unsigned n,
                                              std::vector<Neighbor> &pool) {
  std::ofstream out(std::to_string(n) + ".txt");
  for (size_t i = 0; i < pool.size(); i++) {
    out << pool[i].id << " " << pool[i].distance << std::endl;
  }
  out.close();
}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_,
                    const unsigned traversal_sequence) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  // unsigned range = parameters.Get<unsigned>("R");
  // std::vector<std::mutex> locks(nd_);
  unsigned threads = parameters.Get<unsigned>("T");
  unsigned batch_size = parameters.Get<unsigned>("batchSize");
  // set up a task queue
  std::queue<std::pair<unsigned *, unsigned>> task_queue;
  std::mutex task_queue_mutex;
  std::condition_variable cv;
  unsigned max_queue_size = 10;
  std::thread producer;

  switch (traversal_sequence) {
  case 1:
    // it's bfs
    producer = std::thread([&] {
      bfs bfs(nd_, this->knn);
      while (true) {
        auto buffer = new unsigned[batch_size * threads];

        // std::cout << "Producer is producing tasks" << std::endl;
        const auto count =
            bfs.next(batch_size * threads, this->final_graph_, buffer);
        // generating tasks

        std::unique_lock<std::mutex> lock(task_queue_mutex);
        cv.wait(lock, [&] { return task_queue.size() < max_queue_size; });
        task_queue.push({buffer, count});
        cv.notify_all();
        if (count == 0) {
          // std::cout << "Producer has finished" << std::endl;
          break;
        }
      }
    });
    break;
  case 2:
    // it's dfs
    producer = std::thread([&] {
      dfs dfs(nd_, this->knn);
      while (true) {
        auto buffer = new unsigned[batch_size * threads];

        // std::cout << "Producer is producing tasks" << std::endl;
        const auto count =
            dfs.next(batch_size * threads, this->final_graph_, buffer);
        // generating tasks

        std::unique_lock<std::mutex> lock(task_queue_mutex);
        cv.wait(lock, [&] { return task_queue.size() < max_queue_size; });
        task_queue.push({buffer, count});
        cv.notify_all();

        if (count == 0) {
          // std::cout << "Producer has finished" << std::endl;
          break;
        }
      }
    });
    break;
  case 3:
    // it's bdfs
    producer = std::thread([&] {
      bdfs bdfs(nd_, this->knn, 10, 50);
      while (true) {
        auto buffer = new unsigned[batch_size * threads];

        // std::cout << "Producer is producing tasks" << std::endl;
        const auto count =
            bdfs.next(batch_size * threads, this->final_graph_, buffer);
        // generating tasks

        std::unique_lock<std::mutex> lock(task_queue_mutex);
        cv.wait(lock, [&] { return task_queue.size() < max_queue_size; });
        task_queue.push({buffer, count});
        cv.notify_all();

        if (count == 0) {
          // std::cout << "Producer has finished" << std::endl;
          break;
        }
      }
    });
    break;
  default:
    throw std::invalid_argument("Invalid traversal sequence");
    break;
  }

  std::vector<std::thread> working_threads;
  std::atomic_uint count_working(0);

  std::barrier barrier(threads);
  for (unsigned tid = 0; tid < threads; tid++) {
    working_threads.push_back(std::thread([&, tid] {
      unsigned thread_id = tid;
      // std::cout << "working thread tid: " << thread_id << std::endl;
      std::vector<Neighbor> pool, tmp;
      boost::dynamic_bitset<> flags{nd_, 0};
      while (true) {
        std::unique_lock<std::mutex> lock(task_queue_mutex);
        // std::cout << "TID: " << thread_id << " is waiting for tasks"
        // << std::endl;
        cv.wait(lock, [&] { return !task_queue.empty(); });
        auto [task, count] = task_queue.front();
        if (count == 0) {
          // std::cout << "count is 0, exiting !!!!!!!!!!!!!!" << std::endl;
          lock.unlock();
          break;
        }
        count_working++;
        unsigned real_batch_size = (count + threads - 1) / threads;
        if (count_working == threads) {
          // std::cout << "All threads are working" << std::endl;
          task_queue.pop();
          cv.notify_all();
          count_working = 0;
        }
        lock.unlock();
        // std::cout << "TID: " << thread_id << " is processing a task"
        // << std::endl;
        // start processing the tasks
        auto end_task = task + count;
        auto my_task = task + thread_id * real_batch_size;
        auto alter_size = end_task - my_task;
        real_batch_size =
            alter_size < real_batch_size ? alter_size : real_batch_size;
        for (unsigned i = 0; i < real_batch_size; i++) {
          unsigned n = my_task[i];
          pool.clear();
          tmp.clear();
          flags.reset();
          get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
          sync_prune(n, pool, parameters, flags, cut_graph_);
        }

        barrier.arrive_and_wait();
        // now all task finished, free the memory
        if (thread_id == 0) {
          // std::cout << "All threads finished" << std::endl;
          delete[] task;
        }
      }
    }));
  }

  // join the producer thread
  producer.join();
  // std::cout << "Producer thread finished" << std::endl;
  // join the working threads
  unsigned id = 0;
  for (auto &thread : working_threads) {
    thread.join();
    // std::cout << "Working thread " << id << " finished" << std::endl;
    id++;
  }
  working_threads.clear();
  while (!task_queue.empty()) {
    auto [task, count] = task_queue.front();
    task_queue.pop();
    delete[] task;
  }
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters,
                     const unsigned seq_id) {
  // info("Building the index");
  unsigned range = parameters.Get<unsigned>("R");

  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_, seq_id);
  // std::cout << cut_graph_[100].id;
  // std::cout << cut_graph_[200].id;
  // std::cout << cut_graph_[300].distance;
  // std::cout << cut_graph_[nd_ * (size_t)range - 1].distance;

  //        final_graph_.resize(nd_);
  //
  //        for (size_t i = 0; i < nd_; i++) {
  //            SimpleNeighbor *pool = cut_graph_ + i * (size_t) range;
  //            unsigned pool_size = 0;
  //            for (unsigned j = 0; j < range; j++) {
  //                if (pool[j].distance == -1) break;
  //                pool_size = j;
  //            }
  //            pool_size++;
  //            final_graph_[i].resize(pool_size);
  //            for (unsigned j = 0; j < pool_size; j++) {
  //                final_graph_[i][j] = pool[j].id;
  //            }
  //        }

  //        tree_grow(parameters);

  //        unsigned max = 0, min = 1e6, avg = 0;
  //        for (size_t i = 0; i < nd_; i++) {
  //            auto size = final_graph_[i].size();
  //            max = max < size ? size : max;
  //            min = min > size ? size : min;
  //            avg += size;
  //        }
  //        avg /= 1.0 * nd_;
  //        printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max,
  //        min, avg);
  //
  //        has_built = true;
  delete[] cut_graph_;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id])
      continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare(data_ + dimension_ * id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id])
          continue;
        flags[id] = 1;
        float dist = distance_->compare(query, data_ + dimension_ * id,
                                        (unsigned)dimension_);
        if (dist >= retset[L - 1].distance)
          continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk)
          nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters,
                                  unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_.get();

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
  char *opt_graph = opt_graph_.get();
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id])
      continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_)
      continue;
    _mm_prefetch(opt_graph + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_)
      continue;
    float *x = (float *)(opt_graph + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id])
          continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph + node_size * id);
        float norm = *data;
        data++;
        float dist =
            dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance)
          continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk)
          nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::OptimizeGraph(float *data) { // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_.get();
  node_size = data_len + neighbor_len;
  if (opt_graph_ == nullptr) {
    opt_graph_ = std::unique_ptr<char[]>(new char[node_size * nd_]);
  }
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_.get() + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root,
                   unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root])
    cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty())
        break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_)
    return; // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}

void IndexNSG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    // std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_)
      break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}
} // namespace sjq
