//! the bdfs algorithm

pub fn generate_bdfs(knn_graph: &Vec<Vec<u32>>, max_depth: usize) -> Vec<usize> {
    let total_nodes = knn_graph.len();
    let mut visited_nodes_count = 0;
    let mut visited = vec![false; knn_graph.len()];
    let mut nodes_to_visite = vec![];

    for next_to_visit in 0..total_nodes {
        if visited[next_to_visit] {
            continue;
        }
        visited_nodes_count += explore(
            next_to_visit,
            knn_graph,
            &mut visited,
            &mut nodes_to_visite,
            max_depth,
            0,
        );
        if visited_nodes_count == total_nodes {
            break;
        }
    }
    nodes_to_visite
}
fn explore(
    start_node: usize,
    knn_graph: &Vec<Vec<u32>>,
    visited: &mut Vec<bool>,
    result: &mut Vec<usize>,
    max_depth: usize,
    current_depth: usize,
) -> usize {
    let mut visited_nodes_count = 0;
    if current_depth == max_depth {
        return visited_nodes_count;
    }
    //
    visited[start_node] = true;
    visited_nodes_count += 1;
    result.push(start_node);

    let neighbors = knn_graph[start_node]
        .iter()
        .map(|n| *n as usize)
        .collect::<Vec<_>>();
    for neighbor in neighbors {
        if visited[neighbor] {
            continue;
        }
        visited_nodes_count += explore(
            neighbor,
            knn_graph,
            visited,
            result,
            max_depth,
            current_depth + 1,
        );
    }
    visited_nodes_count
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfs() {
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
            vec![13, 14],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_bdfs(&graph, 2);
        assert_eq!(
            result,
            vec![0, 1, 2, 3, 7, 8, 4, 9, 10, 5, 11, 12, 6, 13, 14]
        );
        // with ilands
        let graph = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
            vec![13, 14],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![16, 17],
            vec![],
            vec![],
        ];
        let result = generate_bdfs(&graph, 2);
        assert_eq!(
            result,
            vec![0, 1, 2, 3, 7, 8, 4, 9, 10, 5, 11, 12, 6, 13, 14, 15, 16, 17]
        );
    }
}
