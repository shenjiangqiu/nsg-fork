use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use tracing::info;

pub struct Fvec {
    pub data: Vec<f32>,
    pub dim: usize,
    pub num: usize,
}

impl Fvec {
    pub fn new(dim: usize, num: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), dim * num);
        Self { data, dim, num }
    }
    pub fn from_file(file_path: &Path) -> Self {
        let mut data = vec![];
        let mut file = File::open(file_path).unwrap();
        let mut k = [0u8; 4];
        file.read_exact(&mut k).unwrap();
        let dim: u32 = u32::from_le_bytes(k);
        info!("dim: {}", dim);
        file.seek(SeekFrom::End(0)).unwrap();
        let fsize = file.stream_position().unwrap() as usize;
        let num = fsize / ((dim as usize + 1) * 4);
        info!("num: {}", num);
        data.reserve(num * dim as usize);
        file.seek(SeekFrom::Start(0)).unwrap();
        for _row_id in 0..num {
            file.seek(SeekFrom::Current(4)).unwrap();
            let mut bytes = vec![0u8; dim as usize * 4];
            file.read_exact(&mut bytes).unwrap();
            let mut buffer = vec![0f32; dim as usize];

            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buffer[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // check the correctness of the fvec file
                assert!(buffer[i].is_finite());
            }

            data.extend(buffer);
        }
        Self {
            data,
            dim: dim as usize,
            num,
        }
    }

    pub fn get_center_point(&self) -> Vec<f32> {
        let mut center = vec![0f32; self.dim];
        for i in 0..self.num {
            for j in 0..self.dim {
                center[j] += self.data[i * self.dim + j];
            }
        }
        for j in 0..self.dim {
            center[j] /= self.num as f32;
        }
        center
    }
    pub fn get_node(&self, node_id: usize) -> &[f32] {
        &self.data[node_id * self.dim..(node_id + 1) * self.dim]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_read_fvec() {
        let fvec = super::Fvec::from_file(std::path::Path::new(
            "/home/sjq/sjqssd/nsg-fork/gist_base.fvecs",
        ));
        assert_eq!(fvec.dim, 960);
        assert_eq!(fvec.num, 1000000);
        assert_eq!(fvec.data.len(), 1000000 * 960);
        for i in 0..10 {
            println!("{}", fvec.data[i])
        }
        for i in ((1000000 - 1) * 960..1000000 * 960).take(10) {
            println!("{}", fvec.data[i])
        }
    }
}
