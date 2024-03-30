use std::fs::File;
use std::io::{prelude::*, SeekFrom};
use std::path::Path;

use tracing::info;

pub struct KnnGraph {
    pub final_graph: Vec<Vec<u32>>,
    pub num: usize,
    pub k: usize,
}

impl KnnGraph {
    pub fn new(final_graph: Vec<Vec<u32>>) -> Self {
        let k = final_graph[0].len();
        let num = final_graph.len();
        assert!(final_graph.iter().all(|x| x.len() == k));
        Self {
            final_graph,
            num,
            k,
        }
    }
    pub fn from_file(filename: &Path) -> Self {
        Self::try_from_file(filename).unwrap()
    }
    pub fn try_from_file(filename: &Path) -> eyre::Result<Self> {
        let mut final_graph = vec![];
        let mut file = File::open(filename)?;
        let mut k = [0u8; 4];
        file.read_exact(&mut k)?;
        let k = u32::from_le_bytes(k);
        info!("k: {}", k);
        file.seek(SeekFrom::End(0))?;
        let fsize = file.stream_position()? as usize;
        let num = fsize / ((k as usize + 1) * 4);
        info!("num: {}", num);
        final_graph.reserve(num);
        file.seek(SeekFrom::Start(0))?;
        for _ in 0..num {
            file.seek(SeekFrom::Current(4))?;
            let mut buffer = vec![0u32; k as usize];
            let mut bytes = vec![0u8; k as usize * 4];
            file.read_exact(&mut bytes)?;
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                let node_id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if node_id >= num as u32 {
                    return Err(eyre::eyre!(
                        "node_id: {},but num is {}, i: {}",
                        node_id,
                        num,
                        i
                    ));
                }
                buffer[i] = node_id;
            }
            final_graph.push(buffer);
        }
        Ok(Self {
            final_graph,
            num,
            k: k as usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{init_logger_debug, knn_graph::KnnGraph};

    #[test]
    #[ignore]
    fn test_load() {
        init_logger_debug();
        let graph_path = "../gist.100nn.graph";
        let knn_graph = KnnGraph::from_file(std::path::Path::new(graph_path));
        assert_eq!(knn_graph.final_graph.len(), 1000000);
        for i in 0..10 {
            println!("{:?}", knn_graph.final_graph[i]);
        }
        for i in 999990..1000000 {
            println!("{:?}", knn_graph.final_graph[i]);
        }
    }
}
