use ndarray::Array1;

/// # Entropy
/// Calculates the empirical entropy of a probability array measured in nats.
///
/// <https://en.wikipedia.org/wiki/Entropy_(information_theory)>
///
/// This assumes that the provided array is the probability (and sums to one).
///
/// The entropy is calculated as follows:
/// ```math
/// H(X) = -Σ p(x) * ln[ p(x) ]
/// ```
///
/// # Usage:
/// ```
/// use ndarray::array;
/// use information::{entropy, prob1d};
///
/// let x = array![0, 0, 1, 1];
/// let p_x = prob1d(&x, 2).unwrap();
/// let h_x = entropy(&p_x);
///
/// assert_eq!(h_x, 0.6931471805599453)
/// ```
///
#[must_use]
pub fn entropy(px: &Array1<f64>) -> f64 {
    (0..px.len()).fold(0.0, |acc, idx| {
        if px[idx] == 0.0 {
            acc
        } else {
            acc - (px[idx] * px[idx].ln())
        }
    })
}

#[cfg(test)]
mod testing {

    use super::entropy;
    use ndarray::array;

    #[test]
    fn test_entropy() {
        // WolframAlpha: Entropy[{0, 0, 1, 1}]
        assert_eq!(entropy(&array![0.5, 0.5]), 0.6931471805599453);

        // WolframAlpha: Entropy[{0, 0, 1, 1, 2, 2}]
        assert_eq!(
            entropy(&array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            1.0986122886681096
        );

        // WolframAlpha: Entropy[{0, 0, 1, 1, 2, 2, 3, 3}]
        assert_eq!(
            entropy(&array![0.25, 0.25, 0.25, 0.25]),
            1.38629436111989061
        );
    }
}
