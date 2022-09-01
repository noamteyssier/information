use ndarray::Array1;
use anyhow::Result;
use crate::prob::prob1d;

/// Calculates the empirical entropy of an integer array
pub fn entropy(x: &Array1<usize>, x_bins: usize) -> Result<f64> {
    let px = prob1d(&x, x_bins)?;
    let info = (0..x_bins)
        .fold(0.0, |acc, idx| {
            if px[idx] == 0.0 {
                acc
            } else {
                acc - (px[idx] * px[idx].ln())
            }
        });
    Ok(info)
}

#[cfg(test)]
mod testing {

    use ndarray::array;
    use super::entropy;

    #[test]
    fn test_entropy_two_values() {
        // prob (0.2, 0.8)
        let x = array![0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let hx = entropy(&x, 2).unwrap();
        assert_eq!(hx, 0.5004024235381879);
    }

    #[test]
    fn test_entropy_three_values() {
        // prob (0.2, 0.2, 0.6)
        let x = array![0, 0, 1, 1, 2, 2, 2, 2, 2, 2];
        let hx = entropy(&x, 3).unwrap();
        assert_eq!(hx, 0.9502705392332347);
    }

    #[test]
    fn test_entropy_zeros() {
        // prob (0.2, 0.2, 0.6, 0.0)
        let x = array![0, 0, 1, 1, 2, 2, 2, 2, 2, 2];
        let hx = entropy(&x, 4).unwrap();
        assert_eq!(hx, 0.9502705392332347);
    }
}
