pub trait DistanceTrait {
    fn distance(left: &[f32], other: &[f32]) -> f32;
}
pub struct L2Distance;
impl DistanceTrait for L2Distance {
    fn distance(left: &[f32], other: &[f32]) -> f32 {
        let mut sum = 0.0;
        left.iter()
            .zip(other.iter())
            .for_each(|(l, r)| sum += (l - r).powi(2));
        sum
    }
}
pub struct L2DistanceAvx512;
impl DistanceTrait for L2DistanceAvx512 {
    fn distance(left: &[f32], other: &[f32]) -> f32 {
        assert_eq!(left.len(), other.len());
        use std::arch::x86_64::*;
        let mut sum = 0.0;
        left.chunks_exact(16)
            .zip(other.chunks_exact(16))
            .for_each(|(l, r)| {
                let l = unsafe { _mm512_loadu_ps(l.as_ptr()) };
                let r = unsafe { _mm512_loadu_ps(r.as_ptr()) };
                let diff = unsafe { _mm512_sub_ps(l, r) };
                let diff = unsafe { _mm512_mul_ps(diff, diff) };
                let diff = unsafe { _mm512_mask_reduce_add_ps(0xFFFF, diff) };
                sum += diff;
            });
        // handle the remaining elements
        let left_len = left.len();
        let remaining = left_len - (left_len / 16) * 16;
        if remaining > 0 {
            let l = &left[(left_len / 16) * 16..];
            let r = &other[(left_len / 16) * 16..];
            let mut diff = [0.0; 16];
            for i in 0..remaining {
                diff[i] = l[i] - r[i];
            }
            let diff = unsafe { _mm512_loadu_ps(diff.as_ptr()) };
            let diff = unsafe { _mm512_mul_ps(diff, diff) };
            let diff = unsafe { _mm512_mask_reduce_add_ps((1 << remaining) - 1, diff) };
            sum += diff;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_l2_distance() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let distance = L2Distance::distance(&left, &right);
        assert_eq!(distance, 27.0);
    }
    #[test]
    fn test_l2_distance_avx512() {
        let left = vec![1.; 960];
        let right = vec![2.; 960];
        let distance = L2DistanceAvx512::distance(&left, &right);
        assert_eq!(distance, L2Distance::distance(&left, &right));
    }
}
