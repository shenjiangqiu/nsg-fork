use bitvec::{bitvec, vec::BitVec};
use rand::Rng;

use crate::{
    distance::{DistanceTrait, L2Distance},
    fvec::Fvec,
    knn_graph::KnnGraph,
    Neighbor, SimpleNeighbor,
};

#[allow(unused)]
pub struct IndexNsg<'a, 'b> {
    knn_graph: &'a KnnGraph,
    fvec: &'b Fvec,
    l: usize,
    r: usize,
    c: usize,
}

impl<'a, 'b> IndexNsg<'a, 'b> {
    pub fn new(knn_graph: &'a KnnGraph, fvec: &'b Fvec, l: usize, r: usize, c: usize) -> Self {
        Self {
            knn_graph,
            fvec,
            l,
            r,
            c,
        }
    }

    pub fn build(
        &self,
        traversal_ordering: &[u32],
        cut_graph: &mut Vec<SimpleNeighbor>,
        center_point_id: u32,
    ) {
        for (index, node_id) in traversal_ordering.iter().enumerate() {
            let mut full_set = vec![];
            let mut ret_set = vec![];
            let mut flags = bitvec!(0; self.knn_graph.num);
            self.get_neighbors(
                center_point_id as usize,
                self.fvec.get_node(*node_id as usize),
                &mut full_set,
                &mut ret_set,
                &mut flags,
                self.l,
            );
            self.sync_prune(*node_id, index, &mut full_set, &mut flags, cut_graph);
        }
    }
    pub fn sync_prune(
        &self,
        query: u32,
        index: usize,
        full_set: &mut Vec<Neighbor>,
        flags: &mut BitVec,
        cut_graph: &mut Vec<SimpleNeighbor>,
    ) {
        for q_neigher in &self.knn_graph.final_graph[query as usize] {
            // put it into pool
            if flags[*q_neigher as usize] {
                continue;
            }
            let left_f = self.fvec.get_node(*q_neigher as usize);
            let distance = L2Distance::distance(left_f, self.fvec.get_node(query as usize));
            full_set.push(Neighbor {
                id: *q_neigher,
                distance,
                flag: true,
            });
        }
        full_set.sort_unstable_by(|x, y| x.distance.partial_cmp(&y.distance).unwrap());
        let mut result = vec![];
        let mut start = 0;
        if full_set[0].id == query {
            start = 1;
        }
        result.push(full_set[start]);
        while result.len() < self.r && (start + 1) < full_set.len() && start < self.c {
            start += 1;
            let p = &full_set[start];
            let mut occlude = false;
            for r in &result {
                if p.id == r.id {
                    occlude = true;
                    break;
                }
                // get the distance between p and r
                let distance = L2Distance::distance(
                    self.fvec.get_node(p.id as usize),
                    self.fvec.get_node(r.id as usize),
                );
                if distance < p.distance {
                    occlude = true;
                    break;
                }
            }
            if !occlude {
                result.push(*p);
            }
        }
        let simple_neighbers = &mut cut_graph[index * self.r..(index + 1) * self.r];
        for (result_neighber, simple_neighbor) in result.into_iter().zip(simple_neighbers) {
            simple_neighbor.id = result_neighber.id;
            simple_neighbor.distance = result_neighber.distance;
        }
    }

    pub fn build_center_point(&self) -> u32 {
        // fist get the center of fvec
        let center_node = self.fvec.get_center_point();
        // get the closest point in knn_graph
        let mut full_set = vec![];
        let mut ret_set = vec![];
        let start = rand::thread_rng().gen_range(0..self.knn_graph.num);
        let mut flags = bitvec!(0; self.knn_graph.num);
        self.get_neighbors(
            start,
            &center_node,
            &mut full_set,
            &mut ret_set,
            &mut flags,
            self.l,
        );
        let center_node = ret_set[0].id;
        // let center_point = self.knn_graph.get_neighbors(&center_f);
        center_node
    }

    /// return ret_set,full_set
    #[allow(unused)]
    pub fn get_neighbors(
        &self,
        start: usize,
        target_node: &[f32],
        full_set: &mut Vec<Neighbor>,
        ret_set: &mut Vec<Neighbor>,
        flags: &mut BitVec,
        l: usize,
    ) {
        let mut init_ids = Vec::with_capacity(l);
        let mut knn_index = 0;
        for (neighber_id, index) in self.knn_graph.final_graph[start].iter().zip(0..l) {
            init_ids.push(*neighber_id);
            flags.set(*neighber_id as usize, true);
        }
        let mut gen = rand::thread_rng();
        while init_ids.len() < l {
            let id = gen.gen_range(0..self.knn_graph.num);
            if flags[id] {
                continue;
            }
            init_ids.push(id as u32);
            flags.set(id, true);
        }
        for id in init_ids {
            let left_f = self.fvec.get_node(id as usize);
            let distance = L2Distance::distance(target_node, left_f);
            full_set.push(Neighbor {
                id,
                distance,
                flag: true,
            });
            ret_set.push(Neighbor {
                id,
                distance,
                flag: true,
            });
        }
        full_set.sort_by(|x, y| x.distance.total_cmp(&y.distance));

        let mut k = 0;
        while k < l {
            // new_k, the new k for next step, will be set if a new neighor inserted into the retset
            let mut nk = l;
            // test k's neighbor, add them into retset
            if ret_set[k].flag {
                ret_set[k].flag = false;
                for neigber in self.knn_graph.final_graph[ret_set[k].id as usize].iter() {
                    if flags[*neigber as usize] {
                        continue;
                    }
                    let left_f = self.fvec.get_node(*neigber as usize);
                    let distance = L2Distance::distance(target_node, left_f);
                    full_set.push(Neighbor {
                        id: *neigber,
                        distance,
                        flag: true,
                    });
                    if distance < ret_set[l - 1].distance {
                        // insert it !
                        let mut i = l - 2;
                        let insertd_place = binary_search_insert(ret_set, *neigber, distance);
                        if insertd_place < nk {
                            nk = insertd_place;
                        }
                    }
                }
            }
            if nk <= k {
                k = nk;
            } else {
                k += 1;
            }
        }
    }
}

/// binary search to insert a new neighbor into the ret_set
///  the ret_set is sorted by distance in ascending order
pub fn binary_search_insert(ret_set: &mut Vec<Neighbor>, id: u32, distance: f32) -> usize {
    let mut left = 0;
    let mut right = ret_set.len() - 1;
    if ret_set[left].distance > distance {
        // insert at the first
        ret_set.pop();

        ret_set.insert(
            0,
            Neighbor {
                id,
                distance,
                flag: true,
            },
        );
        return left;
    }
    if ret_set[right].distance < distance {
        // don't insert
        return right + 1;
    }
    while left < right {
        let mid = (left + right) / 2;
        if ret_set[mid].distance < distance {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    ret_set.pop();
    ret_set.insert(
        left,
        Neighbor {
            id,
            distance,
            flag: true,
        },
    );
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_binary_insert() {
        let mut sorted_array = vec![
            Neighbor {
                id: 1,
                distance: 1.0,
                flag: true,
            },
            Neighbor {
                id: 2,
                distance: 2.0,
                flag: true,
            },
            Neighbor {
                id: 3,
                distance: 3.0,
                flag: true,
            },
        ];
        binary_search_insert(&mut sorted_array, 4, 1.5);
        assert_eq!(
            sorted_array,
            vec![
                Neighbor {
                    id: 1,
                    distance: 1.0,
                    flag: true
                },
                Neighbor {
                    id: 4,
                    distance: 1.5,
                    flag: true
                },
                Neighbor {
                    id: 2,
                    distance: 2.0,
                    flag: true
                }
            ]
        );
        // insert to first
        binary_search_insert(&mut sorted_array, 5, 0.5);
        assert_eq!(
            sorted_array,
            vec![
                Neighbor {
                    id: 5,
                    distance: 0.5,
                    flag: true
                },
                Neighbor {
                    id: 1,
                    distance: 1.0,
                    flag: true
                },
                Neighbor {
                    id: 4,
                    distance: 1.5,
                    flag: true
                }
            ]
        );
        // no insert
        binary_search_insert(&mut sorted_array, 6, 4.5);
        assert_eq!(
            sorted_array,
            vec![
                Neighbor {
                    id: 5,
                    distance: 0.5,
                    flag: true
                },
                Neighbor {
                    id: 1,
                    distance: 1.0,
                    flag: true
                },
                Neighbor {
                    id: 4,
                    distance: 1.5,
                    flag: true
                }
            ]
        );
    }
}
