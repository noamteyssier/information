use ndarray::{Array2, Array3};

/// Measures the joint entropy: H(X,Y)
pub fn joint_entropy(p_xy: &Array2<f64>) -> f64 {
    p_xy.iter()
        .fold(0.0, |acc, p| {
            if *p == 0.0 {
                acc
            } else {
                acc - (p * p.ln())
            }
        })
}

/// Measures the joint entropy: H(X,Y,Z)
pub fn joint_entropy_3d(p_xyz: &Array3<f64>) -> f64 {
    p_xyz.iter()
        .fold(0.0, |acc, xyz| {
            if *xyz == 0.0 {
                acc
            } else {
                acc - (xyz * xyz.ln())
            }
        })
}

#[cfg(test)]
mod testing {

    use ndarray::{array, Array2, Array1, Array3};
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use crate::{entropy::entropy, prob::{prob1d, prob2d}, joint::joint_entropy_3d};
    use super::joint_entropy;

    const N_ITER: usize = 1000;
    const ARRAY_SIZE: usize = 100;

    #[test]
    fn test_joint() {
        let px = array![[0.5, 0.0], [0.25, 0.25]];
        let hx = joint_entropy(&px);
        assert_eq!(hx, 1.0397207708399179);
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Joint_entropy#Nonnegativity
    fn test_nonnegative() {
        for _ in 0..N_ITER {
            let c_xy = Array2::random((2, ARRAY_SIZE), Uniform::new(0.1, 0.8));
            let p_xy = &c_xy / c_xy.sum();
            let h = joint_entropy(&p_xy);
            assert!(h >= 0.0);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Joint_entropy#Nonnegativity
    fn test_nonnegative_3d() {
        for _ in 0..N_ITER {
            let c_xyz = Array3::random((2, 2, ARRAY_SIZE), Uniform::new(0.1, 0.8));
            let p_xyz = &c_xyz / c_xyz.sum();
            let h = joint_entropy_3d(&p_xyz);
            assert!(h >= 0.0);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Joint_entropy#Greater_than_individual_entropies
    fn test_gte_individual_entropies() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_x = prob1d(&c_x, 4).unwrap();
            let p_y = prob1d(&c_y, 4).unwrap();

            let h_xy = joint_entropy(&p_xy);
            let h_x = entropy(&p_x);
            let h_y = entropy(&p_y);

            assert!(h_xy >= h_x);
            assert!(h_xy >= h_y);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Joint_entropy#Less_than_or_equal_to_the_sum_of_individual_entropies
    fn test_lte_individual_entropies() {
        for _ in 0..N_ITER {
            let c_x = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));
            let c_y = Array1::random(ARRAY_SIZE, Uniform::new(0, 3));

            let p_xy = prob2d(&c_x, &c_y, 4, 4).unwrap();
            let p_x = prob1d(&c_x, 4).unwrap();
            let p_y = prob1d(&c_y, 4).unwrap();

            let h_xy = joint_entropy(&p_xy);
            let h_x = entropy(&p_x);
            let h_y = entropy(&p_y);

            assert!(h_xy <= h_x + h_y);
        }
    }

}
