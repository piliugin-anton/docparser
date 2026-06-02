//! PP-DocLayoutV3 image preprocessing (800×800, rescale 1/255, zero mean / unit std).

use anyhow::Result;
use image::{imageops::FilterType, RgbImage};
use ndarray::{Array4, Array2};

pub const TARGET_SIZE: u32 = 800;

#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    pub pixel_values: Array4<f32>,
    pub im_shape: Array2<f32>,
    pub scale_factor: Array2<f32>,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub fn preprocess(image: &RgbImage) -> Result<PreprocessOutput> {
    let (orig_width, orig_height) = image.dimensions();
    let resized = image::imageops::resize(
        image,
        TARGET_SIZE,
        TARGET_SIZE,
        FilterType::Triangle,
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

    let pixel_values = Array4::from_shape_vec((1, 3, TARGET_SIZE as usize, TARGET_SIZE as usize), chw)?;
    let scale_h = TARGET_SIZE as f32 / orig_height as f32;
    let scale_w = TARGET_SIZE as f32 / orig_width as f32;
    let im_shape = Array2::from_shape_vec((1, 2), vec![TARGET_SIZE as f32, TARGET_SIZE as f32])?;
    let scale_factor = Array2::from_shape_vec((1, 2), vec![scale_h, scale_w])?;

    Ok(PreprocessOutput {
        pixel_values,
        im_shape,
        scale_factor,
        orig_width,
        orig_height,
    })
}
