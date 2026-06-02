//! HF processor configs: `generation_config.json`, preprocess, and input id construction.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, DType, Tensor};
use image::RgbImage;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::paddleocr_vl::Config;
use crate::preprocess::{build_input_ids, image_to_pixel_values, num_image_tokens};
use crate::VlmTask;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerationConfig {
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub do_sample: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_max_new_tokens() -> usize {
    4096
}

fn default_temperature() -> f32 {
    1.0
}

impl GenerationConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("generation_config.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        let data = std::fs::read_to_string(&path)?;
        serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))
    }

    pub fn effective_max_new_tokens(&self, requested: usize) -> usize {
        requested.min(self.max_new_tokens)
    }
}

pub struct VlmProcessor {
    pub model_config: Config,
    pub generation: GenerationConfig,
    pub tokenizer: Tokenizer,
}

#[derive(Debug)]
pub struct VlmInputs {
    pub input_ids: Tensor,
    pub input_ids_vec: Vec<u32>,
    pub pixel_values: Tensor,
    pub grid_thw: Tensor,
}

impl VlmProcessor {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let model_config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let generation = GenerationConfig::from_dir(model_dir)?;
        let tokenizer = crate::preprocess::load_tokenizer(model_dir)?;
        Ok(Self {
            model_config,
            generation,
            tokenizer,
        })
    }

    pub fn build_inputs(
        &self,
        image: &RgbImage,
        task: VlmTask,
        device: &Device,
        dtype: DType,
    ) -> Result<VlmInputs> {
        let (pixel_values, grid_thw) = image_to_pixel_values(image, device, dtype)?;
        let spatial_merge = self.model_config.vision_config.spatial_merge_size;
        let n_img = num_image_tokens(&grid_thw, spatial_merge)?;
        let input_ids_vec = self.build_input_ids_vec(task, n_img, device)?;
        let input_ids = Tensor::new(input_ids_vec.as_slice(), device)?.unsqueeze(0)?;
        Ok(VlmInputs {
            input_ids,
            input_ids_vec,
            pixel_values,
            grid_thw,
        })
    }

    fn build_input_ids_vec(
        &self,
        task: VlmTask,
        num_image_tokens: usize,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let t = build_input_ids(
            &self.tokenizer,
            &self.model_config,
            task,
            num_image_tokens,
            device,
        )?;
        Ok(t.to_vec2::<u32>()?[0].clone())
    }
}
