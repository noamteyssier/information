
use ndarray::{Array2, Axis, Zip};

pub fn mutual_information(p_xy: &Array2<f64>) -> f64 {
    Zip::from(p_xy)
        .and_broadcast(&p_xy.sum_axis(Axis(0)))
        .and_broadcast(&p_xy.sum_axis(Axis(1)).insert_axis(Axis(1)))
        .fold(0.0, |acc, xy, x, y| {
            if *xy == 0.0 || *x == 0.0 || *y == 0.0 {
                acc
            } else {
                acc + (xy * (xy / (x * y)).ln())
            }

        })
}

#[cfg(test)]
mod testing {

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, Axis};
    use super::mutual_information;
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use crate::{entropy::entropy, prob::prob2d, joint::joint_entropy, conditional::conditional_entropy};

    const N_ITER: usize = 1000;
    const ARRAY_SIZE: usize = 100;
    const EPSILON: f64 = 1e-14;

    #[test]
    /// https://en.wikipedia.org/wiki/Mutual_information#Nonnegativity
    fn test_nonnegativity() {
        for _ in 0..N_ITER {
            let c_xy = Array2::random((2, ARRAY_SIZE), Uniform::new(0.1, 0.9));
            let p_xy = &c_xy / c_xy.sum();
            let i_xy = mutual_information(&p_xy);

            // Measures: I(X;Y) >= 0
            assert!(i_xy >= 0.0);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Mutual_information#Symmetry
    fn test_symmetry() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_yx = prob2d(&c_y, &c_x, 4, 4).unwrap();

            let i_xy = mutual_information(&p_xy);
            let i_yx = mutual_information(&p_yx);

            // Measures: I(X;Y) = I(Y;X)
            assert_relative_eq!(i_xy, i_yx, epsilon=EPSILON);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
    fn test_properties() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_x = p_xy.sum_axis(Axis(0));
            let p_y = p_xy.sum_axis(Axis(1));

            let i_xy = mutual_information(&p_xy);
            let h_joint_xy = joint_entropy(&p_xy);
            let h_conditional_xy = conditional_entropy(&p_xy);

            let h_x = entropy(&p_x);
            let h_y = entropy(&p_y);

            // Measures: I(X;Y) = H(Y) - H(Y|X)
            assert_relative_eq!(i_xy, h_y - h_conditional_xy, epsilon=EPSILON);
            
            // Measures: I(X:Y) = H(X) + H(Y) - H(X,Y)
            assert_relative_eq!(i_xy, h_x + h_y - h_joint_xy, epsilon=EPSILON);
        }

    }
}
