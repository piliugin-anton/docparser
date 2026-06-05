//! In-tree PaddleOCR-VL Candle inference (vendored from candle-transformers).

use std::path::Path;

use candle_core::{DType, Device};
use image::RgbImage;

use crate::paddleocr_vl::PaddleOCRVLModel;
use crate::preprocess::eos_token_id;
use crate::processor::VlmProcessor;
use crate::{Result, VlmError, VlmTask};

pub struct VlmRunner {
    model: PaddleOCRVLModel,
    processor: VlmProcessor,
    dtype: DType,
    device: Device,
}

impl VlmRunner {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let processor = VlmProcessor::from_dir(model_dir)?;
        let dtype = DType::F32;
        let vb = docparser_candle_utils::var_builder_from_safetensors(model_dir, dtype, &device)?;
        let model = PaddleOCRVLModel::new(&processor.model_config, vb)?;
        Ok(Self {
            model,
            processor,
            dtype,
            device,
        })
    }

    pub fn build_input_ids_vec(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<u32>> {
        let inputs = self
            .processor
            .build_inputs(image, task, &self.device, self.dtype)?;
        Ok(inputs.input_ids_vec)
    }

    pub fn preprocess_grid_thw(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<Vec<u32>>> {
        let inputs = self
            .processor
            .build_inputs(image, task, &self.device, self.dtype)?;
        Ok(inputs.grid_thw.to_vec2()?)
    }

    pub fn generate_token_ids(
        &mut self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let inputs = self
            .processor
            .build_inputs(image, task, &self.device, self.dtype)?;
        let max_new = self
            .processor
            .generation
            .effective_max_new_tokens(max_new_tokens);
        let eos = self.eos_token_id();
        if self.processor.generation.do_sample {
            tracing::warn!("do_sample=true in generation_config; using greedy decode");
        }
        Ok(self.model.generate(
            &inputs.input_ids,
            &inputs.pixel_values,
            &inputs.grid_thw,
            max_new,
            eos,
        )?)
    }

    pub fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let eos = self.eos_token_id();
        let trimmed: Vec<_> = tokens.iter().copied().take_while(|&t| t != eos).collect();
        let text = self
            .processor
            .tokenizer
            .decode(&trimmed, true)
            .map_err(|e| VlmError::Tokenizer(format!("decode: {e}")))?;
        Ok(text.trim().to_string())
    }

    pub fn eos_token_id(&self) -> u32 {
        self.processor
            .generation
            .eos_token_id
            .unwrap_or_else(|| eos_token_id(&self.processor.tokenizer))
    }

    pub fn generate(
        &mut self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<String> {
        let tokens = self.generate_token_ids(image, task, max_new_tokens)?;
        self.decode_token_ids(&tokens)
    }
}
