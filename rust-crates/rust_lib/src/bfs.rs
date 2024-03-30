use std::collections::VecDeque;

use tracing::{info, warn};

pub fn generate_bfs(knn_graph: &Vec<Vec<u32>>) -> Vec<usize> {
    info!("start to generate bfs");
    let total_nodes = knn_graph.len();
    let mut visited_nodes_count = 0;
    let mut visited = vec![false; knn_graph.len()];
    let mut nodes_to_visite = vec![];
    let mut working_queue = VecDeque::new();
    working_queue.push_back(0);
    visited[0] = true;
    visited_nodes_count += 1;
    loop {
        while let Some(node) = working_queue.pop_front() {
            //
            nodes_to_visite.push(node);

            let neighbors = knn_graph[node]
                .iter()
                .map(|n| *n as usize)
                .collect::<Vec<_>>();
            for neighbor in neighbors {
                if visited[neighbor] {
                    continue;
                }
                working_queue.push_back(neighbor);
                visited[neighbor] = true;
                visited_nodes_count += 1;
            }
        }
        if visited_nodes_count == total_nodes {
            break;
        }
        warn!("bfs not finished, finding a new start node");
        // find a node not visited
        let unvisited = visited.iter().position(|&v| !v).unwrap();
        working_queue.push_back(unvisited);
        visited[unvisited] = true;
        visited_nodes_count += 1;
    }
    info!("bfs generated");
    nodes_to_visite
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bfs() {
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_bfs(&graph);
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6]);

        // with ilands
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6, 8],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_bfs(&graph);
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 8, 7]);
    }
}
