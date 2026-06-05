use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use serde::Deserialize;

use crate::grid_sample::upsample_bilinear_align_corners;
use crate::{Result, UvdocError};

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub height: u32,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    /// Resized network input (NCHW BGR, 712×488).
    pub network_input: Tensor,
    /// Full-resolution input (NCHW BGR, float32 in [0, 1]).
    pub original_bgr: Tensor,
}

impl PreprocessorConfig {
    pub fn from_dir(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let data = std::fs::read_to_string(&path)?;
        #[derive(Deserialize)]
        struct Root {
            size: Size,
        }
        #[derive(Deserialize)]
        struct Size {
            height: u32,
            width: u32,
        }
        let root: Root = serde_json::from_str(&data)?;
        Ok(Self {
            height: root.size.height,
            width: root.size.width,
        })
    }
}

/// Matches HF `UVDocImageProcessor`: rescale to [0,1], RGB→BGR, then bilinear resize with `align_corners=true`.
pub fn preprocess(image: &RgbImage, cfg: &PreprocessorConfig, device: &Device) -> Result<Tensor> {
    Ok(preprocess_with_original(image, cfg, device)?.network_input)
}

pub fn preprocess_with_original(
    image: &RgbImage,
    cfg: &PreprocessorConfig,
    device: &Device,
) -> Result<PreprocessOutput> {
    let original_bgr = rgb_to_bgr_tensor(image, device)?;
    let network_input =
        upsample_bilinear_align_corners(&original_bgr, cfg.height as usize, cfg.width as usize)?;
    Ok(PreprocessOutput {
        network_input,
        original_bgr,
    })
}

pub fn rgb_to_bgr_tensor(image: &RgbImage, device: &Device) -> Result<Tensor> {
    let (orig_w, orig_h) = image.dimensions();
    let oh = orig_h as usize;
    let ow = orig_w as usize;
    let mut data = vec![0f32; 3 * oh * ow];
    for y in 0..oh {
        for x in 0..ow {
            let p = image.get_pixel(x as u32, y as u32);
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            data[0 * oh * ow + y * ow + x] = b;
            data[1 * oh * ow + y * ow + x] = g;
            data[2 * oh * ow + y * ow + x] = r;
        }
    }
    Tensor::from_vec(data, (1, 3, oh, ow), device)
        .map_err(|e| UvdocError::Message(format!("original bgr tensor: {e}")))?
        .to_dtype(DType::F32)
        .map_err(UvdocError::Candle)
}
