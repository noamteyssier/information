use ndarray::Array1;
use anyhow::{Result, bail};

/// Calculates the number of events in each bin for a single integer array
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

#[cfg(test)]
mod testing {
    use ndarray::array;
    use super::hist1d;

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
}
