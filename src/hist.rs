use ndarray::{Array1, Array2, Array3};
use anyhow::{Result, bail};

/// Calculates the number of events of each integer bin for a one-dimensional integer array.
///
/// # Usage
/// ```
/// use ndarray::array;
/// use information::hist1d;
///
/// let arr = array![0, 1, 1, 1, 2, 2];
/// let hist = hist1d(&arr, 3).unwrap();
/// assert_eq!(hist, array![1, 3, 2]);
/// ```
pub fn hist1d(arr: &Array1<usize>, nbins: usize) -> Result<Array1<usize>> {
    let mut events = Array1::zeros(nbins);
    for idx in arr.iter() {
        if *idx < nbins {
            events[*idx] += 1;
        }
        else {
            bail!("Out of index error found - raise the number of bins provided");
        }
    }
    Ok(events)
}

/// Calculates the event intersection between two integer arrays of equal size
///
/// # Usage
/// ```
/// use ndarray::array;
/// use information::hist2d;
///
/// let arr_a = array![0, 1, 1, 1, 2, 2];
/// let arr_b = array![1, 0, 0, 1, 2, 3];
/// let expected = array![
///     [0, 1, 0, 0],
///     [2, 1, 0, 0],
///     [0, 0, 1, 1]
/// ];
/// let hist = hist2d(&arr_a, &arr_b, 3, 4).unwrap();
/// assert_eq!(hist.shape(), &[3, 4]);
/// assert_eq!(hist, expected);
/// ```
pub fn hist2d(
    arr_a: &Array1<usize>,
    arr_b: &Array1<usize>,
    nbins_a: usize,
    nbins_b: usize) -> Result<Array2<usize>> 
{
    if arr_a.len() != arr_b.len() {
        bail!("Provided arrays must be of equal size");
    }
    let mut events = Array2::zeros((nbins_a, nbins_b));
    for idx in 0..arr_a.len() {
        let ix = arr_a[idx];
        let jx = arr_b[idx];
        if ix >= nbins_a {
            bail!("Out of index error found - raise the number of bins provided to array 1");
        } else if jx >= nbins_b {
            bail!("Out of index error found - raise the number of bins provided to array 2");
        } else {
            events[(ix, jx)] += 1;
        }
    }
    Ok(events)
}

/// Calculates the event intersection between three integer arrays of equal size
///
/// # Usage
///
/// ```
/// use ndarray::array;
/// use information::hist3d;
///
/// let arr_a = array![0, 1, 1];
/// let arr_b = array![0, 0, 1];
/// let arr_c = array![1, 1, 1];
/// let expected = array![
///     [[0, 1], 
///      [0, 0]],
///
///     [[0, 1], 
///      [0, 1]]
/// ];
/// let hist = hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
/// assert_eq!(hist.shape(), &[2, 2, 2]);
/// assert_eq!(hist, expected);
/// ```
pub fn hist3d(
    arr_a: &Array1<usize>,
    arr_b: &Array1<usize>,
    arr_c: &Array1<usize>,
    nbins_a: usize,
    nbins_b: usize,
    nbins_c: usize) -> Result<Array3<usize>> 
{
    if arr_a.len() != arr_b.len() || arr_a.len() != arr_c.len() {
        bail!("Provided arrays must be of equal size");
    }
    let mut events = Array3::zeros((nbins_a, nbins_b, nbins_c));
    for idx in 0..arr_a.len() {
        let ix = arr_a[idx];
        let jx = arr_b[idx];
        let kx = arr_c[idx];
        if ix >= nbins_a {
            bail!("Out of index error found - raise the number of bins provided to array 1");
        } else if jx >= nbins_b {
            bail!("Out of index error found - raise the number of bins provided to array 2");
        } else if kx >= nbins_c {
            bail!("Out of index error found - raise the number of bins provided to array 2");
        } else {
            events[(ix, jx, kx)] += 1;
        }
    }
    Ok(events)
}

#[cfg(test)]
mod testing {
    use ndarray::array;
    use super::{hist1d, hist2d, hist3d};

    #[test]
    fn test_1d_basic() {
        let arr = array![0, 1, 1, 1, 2, 2];
        let hist = hist1d(&arr, 3).unwrap();
        assert_eq!(hist, array![1, 3, 2]);
    }

    #[test]
    fn test_1d_missing() {
        let arr = array![0, 1, 1, 1, 2, 2];
        let hist = hist1d(&arr, 4).unwrap();
        assert_eq!(hist, array![1, 3, 2, 0]);
    }

    #[test]
    #[should_panic]
    fn test_1d_malform() {
        let arr = array![0, 1, 1, 1, 2, 2];
        hist1d(&arr, 2).unwrap();
    }

    #[test]
    fn test_2d_basic() {
        let arr_a = array![0, 1, 1, 1, 2, 2];
        let arr_b = array![1, 0, 0, 1, 2, 3];
        let expected = array![
            [0, 1, 0, 0],
            [2, 1, 0, 0],
            [0, 0, 1, 1]
        ];
        let hist = hist2d(&arr_a, &arr_b, 3, 4).unwrap();
        assert_eq!(hist.shape(), &[3, 4]);
        assert_eq!(hist, expected);
    }

    #[test]
    fn test_2d_missing() {
        let arr_a = array![0, 1, 1, 1, 2, 2];
        let arr_b = array![1, 0, 0, 1, 2, 3];
        let expected = array![
            [0, 1, 0, 0],
            [2, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]
        ];
        let hist = hist2d(&arr_a, &arr_b, 4, 4).unwrap();
        assert_eq!(hist.shape(), &[4, 4]);
        assert_eq!(hist, expected);
    }

    #[test]
    #[should_panic]
    fn test_2d_malform_a() {
        let arr_a = array![0, 1, 1, 1, 2, 2];
        let arr_b = array![1, 0, 0, 1, 2, 3];
        hist2d(&arr_a, &arr_b, 2, 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_2d_malform_b() {
        let arr_a = array![0, 1, 1, 1, 2, 2];
        let arr_b = array![1, 0, 0, 1, 2, 3];
        hist2d(&arr_a, &arr_b, 3, 3).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_2d_unequal() {
        let arr_a = array![0, 1, 1, 1, 2, 2];
        let arr_b = array![1, 0, 0, 1, 2];
        hist2d(&arr_a, &arr_b, 3, 4).unwrap();
    }

    #[test]
    fn test_3d_basic() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        let expected = array![
            [[0, 1], 
             [0, 0]],

            [[0, 1], 
             [0, 1]]
        ];
        let hist = hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
        assert_eq!(hist.shape(), &[2, 2, 2]);
        assert_eq!(hist, expected);
    }

    #[test]
    fn test_3d_missing() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        let expected = array![
            [[0, 1], 
             [0, 0]],

            [[0, 1], 
             [0, 1]],

            [[0, 0],
             [0, 0]]
        ];
        let hist = hist3d(&arr_a, &arr_b, &arr_c, 3, 2, 2).unwrap();
        assert_eq!(hist.shape(), &[3, 2, 2]);
        assert_eq!(hist, expected);
    }

    #[test]
    #[should_panic]
    fn test_3d_unequal_a() {
        let arr_a = array![0, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
    }


    #[test]
    #[should_panic]
    fn test_3d_unequal_b() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0];
        let arr_c = array![1, 1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_3d_unequal_c() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 2).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_3d_malform_a() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 1, 2, 2).unwrap();
    }


    #[test]
    #[should_panic]
    fn test_3d_malform_b() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 2, 1, 2).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_3d_malform_c() {
        let arr_a = array![0, 1, 1];
        let arr_b = array![0, 0, 1];
        let arr_c = array![1, 1, 1];
        hist3d(&arr_a, &arr_b, &arr_c, 2, 2, 1).unwrap();
    }

}
