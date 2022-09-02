//! # Information
//! This is a crate to perform [information theory](https://en.wikipedia.org/wiki/Information_theory) calculations using [`ndarray`] arrays.
//!
//! ## Entropy Functions
//! * [`entropy()`]
//! * [`joint_entropy!()`]
//! * [`conditional_entropy()`]
//!
//! ## Information Functions
//! * [`mutual_information()`]
//! * [`conditional_mutual_information()`]
//!
//! ## Utility
//! ### `N-d` Histogram
//! * [`hist1d`]
//! * [`hist2d`]
//! * [`hist3d`]
//!
//! ### `N-d` Probability
//! * [`prob1d`]
//! * [`prob2d`]
//! * [`prob3d`]
//!
pub mod hist;
pub mod prob;
pub mod entropy;
pub mod joint;
pub mod conditional;
pub mod mutual;
pub mod cmi;

pub use hist::{hist1d, hist2d, hist3d};
pub use prob::{prob1d, prob2d, prob3d};
pub use entropy::entropy;
pub use conditional::conditional_entropy;
pub use mutual::mutual_information;
pub use cmi::conditional_mutual_information;
