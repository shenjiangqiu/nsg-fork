#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "util.h"
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

namespace sjq_static {
using namespace efanna2e;

class IndexNSG : public Index {
public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m,
                    Index *initializer);

  virtual ~IndexNSG();

  virtual void Save(const char *filename) override;
  virtual void Load(const char *filename) override;

  virtual void Build(size_t n, const float *data, const Parameters &parameters,
                     const unsigned) override;
  void Build_static(size_t n, const float *data, const Parameters &parameters,
                    const unsigned *) ;

  virtual void Search(const float *query, const float *x, size_t k,
                      const Parameters &parameters, unsigned *indices) override;
  void SearchWithOptGraph(const float *query, size_t K,
                          const Parameters &parameters, unsigned *indices);
  void OptimizeGraph(float *data);
  void Load_nn_graph(const char *filename);
  unsigned knn;
public:
  typedef std::vector<std::vector<unsigned>> CompactGraph;
  typedef std::vector<SimpleNeighbors> LockGraph;
  typedef std::vector<nhood> KNNGraph;

  CompactGraph final_graph_;

  Index *initializer_ = nullptr;
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
  void Link_static(const Parameters &parameters, SimpleNeighbor *cut_graph_,
                   const unsigned *traversal_sequence);
  void tree_grow(const Parameters &parameter);
  void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
  void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                const Parameters &parameter);

private:
  unsigned width;
  // end point, should be the center point
  unsigned ep_;
  std::vector<std::mutex> locks;
  char *opt_graph_ = nullptr;
  size_t node_size;
  size_t data_len;
  size_t neighbor_len;
  KNNGraph nnd_graph;
};
} // namespace sjq_static

#endif // EFANNA2E_INDEX_NSG_H
