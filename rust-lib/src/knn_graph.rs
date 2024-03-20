use std::fs::File;
use std::io::{self, prelude::*, SeekFrom};
use std::path::Path;

use tracing::info;

pub struct IndexNSG {
    pub final_graph_: Vec<Vec<u32>>,
}

impl IndexNSG {
    pub fn new() -> IndexNSG {
        IndexNSG {
            final_graph_: Vec::new(),
        }
    }

    pub fn load_nn_graph(&mut self, filename: &Path) -> io::Result<()> {
        self.final_graph_.clear();
        let mut file = File::open(filename)?;
        let mut k = [0u8; 4];
        file.read_exact(&mut k)?;
        let k = u32::from_le_bytes(k);
        info!("k: {}", k);
        file.seek(SeekFrom::End(0))?;
        let fsize = file.stream_position()? as usize;
        let num = fsize / ((k as usize + 1) * 4);
        info!("num: {}", num);
        self.final_graph_.reserve(num);
        file.seek(SeekFrom::Start(0))?;
        for _ in 0..num {
            file.seek(SeekFrom::Current(4))?;
            let mut buffer = vec![0u32; k as usize];
            let mut bytes = vec![0u8; k as usize * 4];
            file.read_exact(&mut bytes)?;
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buffer[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            self.final_graph_.push(buffer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::init_logger_debug;

    #[test]
    #[ignore]
    fn test_load() {
        init_logger_debug();
        let graph_path = "../gist.100nn.graph";
        let mut knn_graph = super::IndexNSG::new();
        knn_graph
            .load_nn_graph(std::path::Path::new(graph_path))
            .unwrap();
        assert_eq!(knn_graph.final_graph_.len(), 1000000);
        for i in 0..10 {
            println!("{:?}", knn_graph.final_graph_[i]);
        }
        for i in 999990..1000000 {
            println!("{:?}", knn_graph.final_graph_[i]);
        }
    }
}
