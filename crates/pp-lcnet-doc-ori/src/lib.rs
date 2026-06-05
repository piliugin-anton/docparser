#![deny(unsafe_code)]

mod config;
mod error;
mod model;
mod nn;
mod preprocess;
mod rotate;

pub use config::PpLcnetConfig;
pub use error::{DocOriError, Result};
pub use model::DocOrientationModel;
pub use preprocess::{preprocess, PreprocessorConfig};
pub use rotate::{rotate_by_angle, rotate_rgb};
