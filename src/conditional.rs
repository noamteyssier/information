use ndarray::{Array2, Axis, Zip};

/// # Conditional Entropy
/// <https://en.wikipedia.org/wiki/Conditional_entropy>
///
/// Calculates the conditional entropy of a random variable `X` given that another random variable
/// `Y` is known measured in nats.
///
/// `H(X|Y)` is calculated as follows:
/// ```math
/// H(X|Y) = - Σ p(x,y) * ln[ p(x,y) / p(x) ]
/// ```
///
/// # Usage
/// ```
/// use ndarray::array;
/// use information::conditional_entropy;
///
/// let p_xy = array![[0.5, 0.0], [0.25, 0.25]];
/// let hx = conditional_entropy(&p_xy);
/// assert_eq!(hx, 0.4773856262211097);
/// ```
pub fn conditional_entropy(p_xy: &Array2<f64>) -> f64 {
    Zip::from(p_xy)
        .and_broadcast(&p_xy.sum_axis(Axis(0)))
        .fold(0.0, |acc, xy, y| {
            if *xy == 0. || *y == 0. {
                acc
            } else {
                acc - (xy * (xy / y).ln())
            }
        })
}

#[cfg(test)]
mod testing {

    use approx::assert_relative_eq;
    use ndarray::{array, Array1};
    use super::conditional_entropy;
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use crate::{entropy::entropy, prob::{prob1d, prob2d}, joint_entropy};

    const N_ITER: usize = 1000;
    const ARRAY_SIZE: usize = 100;
    const EPSILON: f64 = 1e-14;

    #[test]
    fn test_conditional() {
        let p_xy = array![[0.5, 0.0], [0.25, 0.25]];
        let hx = conditional_entropy(&p_xy);
        assert_eq!(hx, 0.4773856262211097);
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Conditional_entropy#Conditional_entropy_equals_zero
    fn test_conditional_zero_case() {
        let p_xy = array![[0., 0.], [0., 0.]];
        let h_xy = conditional_entropy(&p_xy);
        assert_eq!(h_xy, 0.0);
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Conditional_entropy#Chain_rule
    fn test_chain_rule() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_yx = prob2d(&c_y, &c_x, 4, 4).unwrap();
            let p_x = prob1d(&c_x, 4).unwrap();
            let p_y = prob1d(&c_y, 4).unwrap();

            let h_conditional_xy = conditional_entropy(&p_xy);
            let h_conditional_yx = conditional_entropy(&p_yx);
            let h_joint_xy = joint_entropy!(&p_xy);
            let h_x = entropy(&p_x);
            let h_y = entropy(&p_y);

            // measures H(X|Y) = H(X,Y) - H(Y)
            assert_relative_eq!(h_conditional_xy, h_joint_xy - h_y, epsilon = EPSILON);

            // measures H(Y|X) = H(X,Y) - H(X)
            assert_relative_eq!(h_conditional_yx, h_joint_xy - h_x, epsilon = EPSILON);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Conditional_entropy#Bayes'_rule
    fn test_bayes_rule() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_yx = prob2d(&c_y, &c_x, 4, 4).unwrap();
            let p_x = prob1d(&c_x, 4).unwrap();
            let p_y = prob1d(&c_y, 4).unwrap();

            let h_conditional_xy = conditional_entropy(&p_xy);
            let h_conditional_yx = conditional_entropy(&p_yx);
            let h_x = entropy(&p_x);
            let h_y = entropy(&p_y);

            // Measures: H(Y|X) = H(X|Y) - H(X) + H(Y)
            assert_relative_eq!(h_conditional_yx, h_conditional_xy - h_x + h_y, epsilon = EPSILON);
        }
    }

}
