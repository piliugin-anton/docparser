//! Layout detection entry point (native Candle).

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::RgbImage;

use crate::image_processor::LayoutImageProcessor;
use crate::postprocess::post_process_object_detection;
use crate::preprocess::preprocess;
use crate::pp_doclayout_v3::{PpDocLayoutV3Config, PpDocLayoutV3ForObjectDetection};
use crate::LayoutElement;

pub struct LayoutRunner {
    model: PpDocLayoutV3ForObjectDetection,
    device: Device,
    id2label: HashMap<u32, String>,
    image_processor: LayoutImageProcessor,
    detection_threshold: f32,
}

impl LayoutRunner {
    pub fn load(model_dir: &Path, detection_threshold: f32) -> Result<Self> {
        let device = Device::Cpu;
        let cfg = PpDocLayoutV3Config::from_dir(model_dir)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let model = PpDocLayoutV3ForObjectDetection::load(model_dir, &device)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let image_processor = LayoutImageProcessor::from_dir(model_dir)?;
        Ok(Self {
            id2label: cfg.id2label_map(),
            model,
            device,
            image_processor,
            detection_threshold,
        })
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        let prep = preprocess(image, &self.device, &self.image_processor)?;
        let pixel_mask = Tensor::ones(
            (1, prep.pixel_values.dims()[2], prep.pixel_values.dims()[3]),
            prep.pixel_values.dtype(),
            &self.device,
        )?;
        let outputs = self
            .model
            .forward(&prep.pixel_values, &pixel_mask)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        post_process_object_detection(
            &outputs,
            prep.orig_height,
            prep.orig_width,
            &self.id2label,
            self.detection_threshold,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
