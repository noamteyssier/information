use ndarray::{Array3, Axis, Zip};


pub fn conditional_mutual_information(p_xyz: &Array3<f64>) -> f64 {
    let p_xz = p_xyz.sum_axis(Axis(1));
    let p_yz = p_xyz.sum_axis(Axis(0));
    let p_z = p_xz.sum_axis(Axis(0));

    Zip::from(p_xyz)
        .and_broadcast(&p_xz.insert_axis(Axis(1)))
        .and_broadcast(&p_yz)
        .and_broadcast(&p_z)
        .fold(0.0, |acc, xyz, xz, yz, z| {
            if *xyz == 0.0 || *xz == 0.0 || *yz == 0.0 || *z == 0.0 {
                acc
            }
            else {
                // println!(">> {} {} {} {}", xyz, xz, yz, z);
                acc + (xyz * ((z * xyz) / (xz * yz)).ln())
            }
        })
}

#[cfg(test)]
mod testing {

    use approx::assert_relative_eq;
    use ndarray::Array1;
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use crate::{entropy::entropy, prob::{prob1d, prob2d, prob3d}, joint::{joint_entropy, joint_entropy_3d}};
    use super::conditional_mutual_information;

    const N_ITER: usize = 1000;
    const ARRAY_SIZE: usize = 100;
    const EPSILON: f64 = 1e-12;

    #[test]
    /// https://en.wikipedia.org/wiki/Conditional_mutual_information#Nonnegativity
    fn test_nonnegative() {
        for _ in 0..N_ITER {
            let x = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));
            let y = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));
            let z = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));
            let xyz = prob3d(&x, &y, &z, 2, 2, 2).unwrap();

            let cmi = conditional_mutual_information(&xyz);

            // Measures: I(X;Y|Z) >= 0
            assert!(cmi >= 0.0);
        }
    }

    #[test]
    /// https://en.wikipedia.org/wiki/Conditional_mutual_information#Some_identities
    fn test_identity() {
        for _ in 0..N_ITER {
            let x = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));
            let y = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));
            let z = Array1::random(ARRAY_SIZE, Uniform::new(0, 2));

            let p_xyz = prob3d(&x, &y, &z, 2, 2, 2).unwrap();
            let p_xz = prob2d(&x, &z, 2, 2).unwrap();
            let p_yz = prob2d(&y, &z, 2, 2).unwrap();
            let p_z = prob1d(&z, 2).unwrap();

            let i_xyz = conditional_mutual_information(&p_xyz);
            let h_xz = joint_entropy(&p_xz);
            let h_yz = joint_entropy(&p_yz);
            let h_xyz = joint_entropy_3d(&p_xyz);
            let h_z = entropy(&p_z);

            // Measures: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
            assert_relative_eq!(i_xyz, h_xz + h_yz - h_xyz - h_z, epsilon=EPSILON);
        }

    }

}
