//! Layout detection entry point (native Candle).

use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::RgbImage;

use crate::postprocess::post_process_object_detection;
use crate::preprocess::preprocess;
use crate::pp_doclayout_v3::PpDocLayoutV3ForObjectDetection;
use crate::LayoutElement;

pub struct LayoutRunner {
    model: PpDocLayoutV3ForObjectDetection,
    device: Device,
}

impl LayoutRunner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let model = PpDocLayoutV3ForObjectDetection::load(model_dir, &device)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { model, device })
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        let prep = preprocess(image, &self.device)?;
        let (_h, w) = (prep.orig_height, prep.orig_width);
        let pixel_mask = Tensor::ones(
            (1, prep.pixel_values.dims()[2], prep.pixel_values.dims()[3]),
            prep.pixel_values.dtype(),
            &self.device,
        )?;
        let outputs = self
            .model
            .forward(&prep.pixel_values, &pixel_mask)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        post_process_object_detection(&outputs, prep.orig_height, prep.orig_width)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

pub fn detect(model_dir: &Path, image: &RgbImage) -> Result<Vec<LayoutElement>> {
    let runner = LayoutRunner::load(model_dir)?;
    runner.detect(image)
}
