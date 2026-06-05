//! PP-DocLayoutV3 image preprocessing via HF `preprocessor_config.json`.

use candle_core::{Device, Tensor};
use image::RgbImage;

use crate::Result;
use crate::image_processor::LayoutImageProcessor;

#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    pub pixel_values: Tensor,
    pub im_shape: Tensor,
    pub scale_factor: Tensor,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub fn preprocess(
    image: &RgbImage,
    device: &Device,
    processor: &LayoutImageProcessor,
) -> Result<PreprocessOutput> {
    processor.preprocess(image, device)
}
