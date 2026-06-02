//! Native Candle implementation of PP-DocLayoutV3.

mod config;
mod decoder;
mod deformable;
mod detection;
mod encoder;
mod global_pointer;
mod grid_sample;
mod hgnet_v2;
mod model;
mod nn;
pub(crate) mod ops;

pub use config::{HgNetV2Config, PpDocLayoutV3Config};
pub use detection::PpDocLayoutV3ForObjectDetection;
pub use model::ModelOutputs;
