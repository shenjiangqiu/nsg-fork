#ifndef INDEX_SJQ_H
#define INDEX_SJQ_H
#include "distance.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "util.h"
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

namespace sjq_async {
using namespace efanna2e;

class IndexNSG {
public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                    std::unique_ptr<Index> &&initializer);

  ~IndexNSG();

  void Save(const char *filename);
  void Load(const char *filename);

  void Build(size_t n, const float *data, const Parameters &parameters,
             const unsigned);

  void Search(const float *query, const float *x, size_t k,
              const Parameters &parameters, unsigned *indices);
  void SearchWithOptGraph(const float *query, size_t K,
                          const Parameters &parameters, unsigned *indices);
  void OptimizeGraph(float *data);
  void Load_nn_graph(const char *filename);
  inline bool HasBuilt() const { return has_built; }

  inline size_t GetDimension() const { return dimension_; };

  inline size_t GetSizeOfDataset() const { return nd_; }

  inline const float *GetDataset() const { return data_; }

protected:
  typedef std::vector<std::vector<unsigned>> CompactGraph;
  typedef std::vector<SimpleNeighbors> LockGraph;
  typedef std::vector<nhood> KNNGraph;

  void init_graph(const Parameters &parameters);
  void get_neighbors(const float *query, const Parameters &parameter,
                     std::vector<Neighbor> &retset,
                     std::vector<Neighbor> &fullset);
  void get_neighbors(const float *query, const Parameters &parameter,
                     boost::dynamic_bitset<> &flags,
                     std::vector<Neighbor> &retset,
                     std::vector<Neighbor> &fullset);
  // void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph&
  // cut_graph_);
  void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                   SimpleNeighbor *cut_graph_);
  void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                  const Parameters &parameter, boost::dynamic_bitset<> &flags,
                  SimpleNeighbor *cut_graph_);
  void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_,
            const unsigned traversal_sequence);
  void tree_grow(const Parameters &parameter);
  void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
  void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                const Parameters &parameter);

  CompactGraph final_graph_;

  std::unique_ptr<Index> initializer_;

private:
  unsigned width;
  // end point, should be the center point
  unsigned ep_;
  std::vector<std::mutex> locks;
  std::unique_ptr<char[]> opt_graph_;
  size_t node_size;
  size_t data_len;
  size_t neighbor_len;
  KNNGraph nnd_graph;

  // from index.h
  const size_t dimension_;

    // the raw fvec data
  const float *data_ = nullptr;
  // the number of nodes
  size_t nd_;
  std::unique_ptr<efanna2e::Distance> distance_;
  bool has_built;

  // the number of neighbors
  unsigned knn;
};

void load_nn_graph(const char *filename,
                   std::vector<std::vector<unsigned>> &final_graph_,
                   unsigned &dim, unsigned &num);
} // namespace sjq

#endif // EFANNA2E_INDEX_NSG_H
