use ndarray::{Array1, Array2, Array3};
use anyhow::Result;
use crate::hist::{hist1d, hist2d, hist3d};

/// Calculates the probability of events in each bin for a single integer array
///
/// # Usage
///
/// ```
/// use ndarray::{array, Array1};
/// use information::prob1d; 
///
/// let arr = array![0, 1, 2];
/// let expected = Array1::from_elem(3, 1.0 / 3.0);
/// let prob = prob1d(&arr, 3).unwrap();
/// assert_eq!(prob, expected);
/// ```
pub fn prob1d(arr: &Array1<usize>, nbins: usize) -> Result<Array1<f64>> {
    let hist = hist1d(arr, nbins)?.mapv(|x| x as f64);
    let total = hist.sum();
    Ok(hist / total)
}

/// Calculates the event intersection probability between two arrays of equal size
///
/// # Usage
/// ```
/// use ndarray::array;
/// use information::prob2d;
///
/// let arr_a = array![0, 1];
/// let arr_b = array![0, 1];
/// let expected = array![[0.5, 0.0],
///                       [0.0, 0.5]];
/// let prob = prob2d(&arr_a, &arr_b, 2, 2).unwrap();
/// assert_eq!(prob.shape(), &[2, 2]);
/// assert_eq!(prob, expected);
/// ```
pub fn prob2d(
    arr_a: &Array1<usize>,
    arr_b: &Array1<usize>,
    nbins_a: usize,
    nbins_b: usize) -> Result<Array2<f64>> 
{
    let hist = hist2d(arr_a, arr_b, nbins_a, nbins_b)?.mapv(|x| x as f64);
    let total = hist.sum();
    Ok(hist / total)
}

/// Calculates the event intersection probability between three arrays of equal size
///
/// # Usage
/// ```
/// use ndarray::array;
/// use information::prob3d;
///
/// let arr_a = array![0, 1];
/// let arr_b = array![0, 1];
/// let arr_c = array![0, 1];
/// let expected = array![[[0.5, 0.0], [0.0, 0.0]],
///                       [[0.0, 0.0], [0.0, 0.5]]];
/// let prob = prob3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
/// assert_eq!(prob.shape(), &[2, 2, 2]);
/// assert_eq!(prob, expected);
/// ```
pub fn prob3d(
    arr_a: &Array1<usize>,
    arr_b: &Array1<usize>,
    arr_c: &Array1<usize>,
    nbins_a: usize,
    nbins_b: usize,
    nbins_c: usize) -> Result<Array3<f64>> 
{
    let hist = hist3d(arr_a, arr_b, arr_c, nbins_a, nbins_b, nbins_c)?.mapv(|x| x as f64);
    let total = hist.sum();
    Ok(hist / total)
}

#[cfg(test)]
mod testing {
    use ndarray::{array, Array1};
    use super::{prob1d, prob2d, prob3d};

    #[test]
    fn test_1d_basic() {
        let arr = array![0, 1, 2];
        let expected = Array1::from_elem(3, 1.0 / 3.0);
        let prob = prob1d(&arr, 3).unwrap();
        assert_eq!(prob, expected);
    }

    #[test]
    fn test_1d_missing() {
        let arr = array![0, 1, 2];
        let mut expected = Array1::from_elem(4, 1.0 / 3.0);
        expected[3] = 0.0;
        let prob = prob1d(&arr, 4).unwrap();
        assert_eq!(prob, expected);
    }

    #[test]
    fn test_2d_basic() {
        let arr_a = array![0, 1];
        let arr_b = array![0, 1];
        let expected = array![
            [0.5, 0.0],
            [0.0, 0.5]
        ];
        let prob = prob2d(&arr_a, &arr_b, 2, 2).unwrap();
        assert_eq!(prob.shape(), &[2, 2]);
        assert_eq!(prob, expected);
    }

    #[test]
    fn test_2d_missing() {
        let arr_a = array![0, 1];
        let arr_b = array![0, 1];
        let expected = array![
            [0.5, 0.0],
            [0.0, 0.5],
            [0.0, 0.0]
        ];
        let prob = prob2d(&arr_a, &arr_b, 3, 2).unwrap();
        assert_eq!(prob.shape(), &[3, 2]);
        assert_eq!(prob, expected);
    }

    #[test]
    fn test_3d_basic() {
        let arr_a = array![0, 1];
        let arr_b = array![0, 1];
        let arr_c = array![0, 1];
        let expected = array![
            [[0.5, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.5]]
        ];
        let prob = prob3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
        assert_eq!(prob.shape(), &[2, 2, 2]);
        assert_eq!(prob, expected);
    }

}
