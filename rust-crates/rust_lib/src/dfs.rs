use tracing::{debug, info};

pub fn generate_dfs(knn_graph: &Vec<Vec<u32>>) -> Vec<usize> {
    info!("start to generate dfs");
    let total_nodes = knn_graph.len();
    let mut visited_nodes_count = 0;
    let mut visited = vec![false; knn_graph.len()];
    let mut nodes_to_visite = vec![];
    let mut working_stack = vec![];

    for next in 0..total_nodes {
        if visited[next] {
            continue;
        }
        debug!("next: {}", next);
        working_stack.push(next);
        visited[next] = true;
        visited_nodes_count += 1;

        while let Some(node) = working_stack.pop() {
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
                working_stack.push(neighbor);
                visited[neighbor] = true;
                visited_nodes_count += 1;
            }
        }
        if visited_nodes_count == total_nodes {
            break;
        }
        // find a node not visited
        debug!(
            "dfs not finished, finding a new start node {}/{}",
            next, total_nodes
        );
    }
    info!("dfs generated");
    nodes_to_visite
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
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let result = generate_dfs(&graph);
        assert_eq!(result, vec![0, 2, 6, 5, 1, 4, 3]);

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
        let result = generate_dfs(&graph);
        assert_eq!(result, vec![0, 2, 8, 6, 5, 1, 4, 3, 7]);
    }
}
