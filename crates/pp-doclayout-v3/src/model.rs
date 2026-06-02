//! Candle port of PP-DocLayoutV3 (transformers modeling_pp_doclayout_v3.py).
//!
//! Inference is not implemented yet; this module is the integration point for the port.

use std::path::Path;

use anyhow::bail;
use image::RgbImage;

use crate::{LayoutElement, list_safetensor_keys};

pub fn detect(model_dir: &Path, image: &RgbImage) -> Result<Vec<LayoutElement>, anyhow::Error> {
    let _ = image;
    let _keys = list_safetensor_keys(model_dir)?;
    bail!(
        "PP-DocLayoutV3 Candle inference is not implemented yet ({} tensor keys loaded from HF safetensors)",
        _keys.len()
    )
}
