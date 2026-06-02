//! Object-detection wrapper (last decoder layer outputs).

use std::path::Path;

use candle_core::{Device, DType};
use docparser_candle_utils::var_builder_from_safetensors;

use super::config::PpDocLayoutV3Config;
use super::model::{ModelOutputs, PpDocLayoutV3Model};

pub struct PpDocLayoutV3ForObjectDetection {
    model: PpDocLayoutV3Model,
}

impl PpDocLayoutV3ForObjectDetection {
    pub fn load(model_dir: &Path, device: &Device) -> candle_core::Result<Self> {
        let cfg = PpDocLayoutV3Config::from_dir(model_dir)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vb = var_builder_from_safetensors(model_dir, DType::F32, device)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .pp("model");
        let model = PpDocLayoutV3Model::load(&cfg, vb)?;
        Ok(Self { model })
    }

    pub fn forward(&self, pixel_values: &candle_core::Tensor, pixel_mask: &candle_core::Tensor) -> candle_core::Result<ModelOutputs> {
        self.model.forward(pixel_values, pixel_mask)
    }
}
