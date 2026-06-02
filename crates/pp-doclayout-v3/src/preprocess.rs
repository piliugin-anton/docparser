//! PP-DocLayoutV3 image preprocessing (800×800, rescale 1/255, zero mean / unit std).

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, RgbImage};

pub const TARGET_SIZE: u32 = 800;

#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    pub pixel_values: Tensor,
    pub im_shape: Tensor,
    pub scale_factor: Tensor,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub fn preprocess(image: &RgbImage, device: &Device) -> Result<PreprocessOutput> {
    let (orig_width, orig_height) = image.dimensions();
    // Bicubic (CatmullRom); close to HF torchvision bicubic. Borderline scores on tiny images
    // are handled by the pipeline full-image fallback when no layout boxes pass threshold.
    let resized = image::imageops::resize(
        image,
        TARGET_SIZE,
        TARGET_SIZE,
        FilterType::CatmullRom,
    );

    let mut chw = vec![0f32; (3 * TARGET_SIZE * TARGET_SIZE) as usize];
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let p = resized.get_pixel(x, y);
            for c in 0..3 {
                let idx = (c * TARGET_SIZE * TARGET_SIZE + y * TARGET_SIZE + x) as usize;
                chw[idx] = p[c as usize] as f32 / 255.0;
            }
        }
    }

    let pixel_values = Tensor::from_vec(
        chw,
        (1, 3, TARGET_SIZE as usize, TARGET_SIZE as usize),
        device,
    )?;
    let scale_h = TARGET_SIZE as f32 / orig_height as f32;
    let scale_w = TARGET_SIZE as f32 / orig_width as f32;
    let im_shape = Tensor::new(&[TARGET_SIZE as f32, TARGET_SIZE as f32], device)?.reshape((1, 2))?;
    let scale_factor = Tensor::new(&[scale_h, scale_w], device)?.reshape((1, 2))?;

    Ok(PreprocessOutput {
        pixel_values,
        im_shape,
        scale_factor,
        orig_width,
        orig_height,
    })
}
