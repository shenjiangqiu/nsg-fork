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
