use ndarray::Array1;

/// Calculates the empirical entropy of a probability
pub fn entropy(px: &Array1<f64>) -> f64 {
    (0..px.len())
        .fold(0.0, |acc, idx| {
            if px[idx] == 0.0 {
                acc
            } else {
                acc - (px[idx] * px[idx].ln())
            }
        })
}

#[cfg(test)]
mod testing {

    use ndarray::array;
    use super::entropy;

    #[test]
    fn test_entropy() {

        // WolframAlpha: Entropy[{0, 0, 1, 1}]
        assert_eq!(entropy(&array![0.5, 0.5]), 0.6931471805599453);

        // WolframAlpha: Entropy[{0, 0, 1, 1, 2, 2}]
        assert_eq!(entropy(&array![1.0/3.0, 1.0/3.0, 1.0/3.0]), 1.0986122886681096);

        // WolframAlpha: Entropy[{0, 0, 1, 1, 2, 2, 3, 3}]
        assert_eq!(entropy(&array![0.25, 0.25, 0.25, 0.25]), 1.38629436111989061);
    }

}
