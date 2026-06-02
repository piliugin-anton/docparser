//! Candle port of PaddleOCR-VL-1.6 (HF modeling_paddleocr_vl.py).
//!
//! Inference is not implemented yet; this module is the integration point for the port.

use std::path::Path;

use anyhow::bail;
use candle_core::Device;
use image::RgbImage;

use crate::{VlmTask, list_safetensor_keys};

pub fn generate(
    model_dir: &Path,
    device: &Device,
    image: &RgbImage,
    task: VlmTask,
    max_new_tokens: usize,
) -> Result<String, anyhow::Error> {
    let _ = (device, image, task, max_new_tokens);
    let _keys = list_safetensor_keys(model_dir)?;
    bail!(
        "PaddleOCR-VL Candle inference is not implemented yet ({} tensor keys loaded from HF safetensors)",
        _keys.len()
    )
}
