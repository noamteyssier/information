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
pub mod cmi;
pub mod conditional;
pub mod entropy;
pub mod hist;
pub mod joint;
pub mod mutual;
pub mod prob;

pub use cmi::conditional_mutual_information;
pub use conditional::conditional_entropy;
pub use entropy::entropy;
pub use hist::{hist1d, hist2d, hist3d};
pub use mutual::mutual_information;
pub use prob::{prob1d, prob2d, prob3d};
