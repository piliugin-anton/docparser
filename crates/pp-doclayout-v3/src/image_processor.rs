//! HF `PPDocLayoutV3ImageProcessor` (from `preprocessor_config.json`).

use std::path::Path;

use candle_core::{Device, Tensor};
use fast_image_resize::images::Image;
use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::RgbImage;
use serde::Deserialize;

use crate::preprocess::PreprocessOutput;
use crate::{LayoutError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutPreprocessorConfig {
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub size: Size,
    pub resample: u32,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Size {
    pub height: u32,
    pub width: u32,
}

impl LayoutPreprocessorConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        let data = std::fs::read_to_string(&path)
            .map_err(|e| LayoutError::Message(format!("read {}: {e}", path.display())))?;
        serde_json::from_str(&data)
            .map_err(|e| LayoutError::Message(format!("parse {}: {e}", path.display())))
    }
}

pub struct LayoutImageProcessor {
    cfg: LayoutPreprocessorConfig,
}

impl LayoutImageProcessor {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            cfg: LayoutPreprocessorConfig::from_dir(model_dir)?,
        })
    }

    pub fn preprocess(&self, image: &RgbImage, device: &Device) -> Result<PreprocessOutput> {
        let (orig_width, orig_height) = image.dimensions();
        let target_w = self.cfg.size.width;
        let target_h = self.cfg.size.height;

        let resized = if self.cfg.do_resize {
            resize_bicubic(image, target_w, target_h)?
        } else {
            image.clone()
        };

        let th = target_h as usize;
        let tw = target_w as usize;
        let mut chw = vec![0f32; 3 * th * tw];
        for y in 0..target_h {
            for x in 0..target_w {
                let p = resized.get_pixel(x, y);
                for c in 0..3 {
                    let mut v = p[c] as f32;
                    if self.cfg.do_rescale {
                        v *= self.cfg.rescale_factor;
                    }
                    if self.cfg.do_normalize {
                        v = (v - self.cfg.image_mean[c]) / self.cfg.image_std[c];
                    }
                    let idx = c * th * tw + y as usize * tw + x as usize;
                    chw[idx] = v;
                }
            }
        }

        let pixel_values =
            Tensor::from_vec(chw, (1, 3, target_h as usize, target_w as usize), device)?;
        let scale_h = target_h as f32 / orig_height as f32;
        let scale_w = target_w as f32 / orig_width as f32;
        let im_shape = Tensor::new(&[target_h as f32, target_w as f32], device)?.reshape((1, 2))?;
        let scale_factor = Tensor::new(&[scale_h, scale_w], device)?.reshape((1, 2))?;

        Ok(PreprocessOutput {
            pixel_values,
            im_shape,
            scale_factor,
            orig_width,
            orig_height,
        })
    }
}

fn resize_bicubic(image: &RgbImage, target_w: u32, target_h: u32) -> Result<RgbImage> {
    let (src_w, src_h) = image.dimensions();
    let src_bytes = image.as_raw();
    let src_img = Image::from_vec_u8(src_w, src_h, src_bytes.clone(), PixelType::U8x3)
        .map_err(|e| LayoutError::ImageResize(format!("fast_image_resize src: {e}")))?;
    let mut dst_img = Image::new(target_w, target_h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    resizer
        .resize(
            &src_img,
            &mut dst_img,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
                fast_image_resize::FilterType::CatmullRom,
            )),
        )
        .map_err(|e| LayoutError::ImageResize(format!("resize: {e}")))?;
    RgbImage::from_raw(target_w, target_h, dst_img.into_vec())
        .ok_or_else(|| LayoutError::ImageResize("invalid resized image buffer".into()))
}
