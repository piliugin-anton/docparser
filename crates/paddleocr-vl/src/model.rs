//! In-tree PaddleOCR-VL Candle inference (vendored from candle-transformers).

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use image::RgbImage;
use tokenizers::Tokenizer;

use crate::paddleocr_vl::{Config, PaddleOCRVLModel};
use crate::preprocess::{
    build_input_ids, eos_token_id, image_to_pixel_values, load_tokenizer, num_image_tokens,
};
use crate::VlmTask;

pub struct VlmRunner {
    model: PaddleOCRVLModel,
    tokenizer: Tokenizer,
    config: Config,
    dtype: DType,
    device: Device,
}

impl VlmRunner {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let dtype = DType::F32;
        let vb = docparser_candle_utils::var_builder_from_safetensors(model_dir, dtype, &device)?;
        let model = PaddleOCRVLModel::new(&config, vb)?;
        let tokenizer = load_tokenizer(model_dir)?;
        Ok(Self {
            model,
            tokenizer,
            config,
            dtype,
            device,
        })
    }

    pub fn generate(
        &mut self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<String> {
        let (pixel_values, grid_thw) = image_to_pixel_values(image, &self.device, self.dtype)?;
        let spatial_merge = self.config.vision_config.spatial_merge_size;
        let n_img = num_image_tokens(&grid_thw, spatial_merge)?;
        let input_ids = build_input_ids(
            &self.tokenizer,
            &self.config,
            task,
            n_img,
            &self.device,
        )?;
        let eos = eos_token_id(&self.tokenizer);
        let tokens = self.model.generate(
            &input_ids,
            &pixel_values,
            &grid_thw,
            max_new_tokens,
            eos,
        )?;
        let text = self
            .tokenizer
            .decode(
                &tokens.into_iter().take_while(|&t| t != eos).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(text.trim().to_string())
    }
}

pub fn generate(
    model_dir: &Path,
    device: &Device,
    image: &RgbImage,
    task: VlmTask,
    max_new_tokens: usize,
) -> Result<String> {
    let mut runner = VlmRunner::load(model_dir, device.clone())?;
    runner.generate(image, task, max_new_tokens)
}
