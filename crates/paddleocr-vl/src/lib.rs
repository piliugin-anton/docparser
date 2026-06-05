#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

mod error;

pub use error::{Result, VlmError};
use anyhow::Context;
use candle_core::Device;
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};

mod model;
pub mod paddleocr_vl;
mod preprocess;
pub mod processor;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VlmTask {
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

impl VlmTask {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Chart => "Chart Recognition:",
            Self::Spotting => "Spotting:",
            Self::Seal => "Seal Recognition:",
        }
    }
}

/// Maps a PP-DocLayoutV3 region label to the VLM prompt task.
///
/// See `docs/layout_labels_and_models.md` in the repo root for the full label list.
pub fn task_for_layout_label(label: &str) -> VlmTask {
    match label {
        "table" => VlmTask::Table,
        "display_formula" | "inline_formula" | "formula" => VlmTask::Formula,
        "chart" => VlmTask::Chart,
        "seal" => VlmTask::Seal,
        "vertical_text" | "text" | "text_block" | "content" | "doc_title" | "paragraph_title"
        | "abstract" | "reference" | "reference_content" | "figure_title" | "algorithm"
        | "vision_footnote" | "number" | "footnote" | "formula_number" => VlmTask::Ocr,
        "image" | "header_image" | "footer_image" => VlmTask::Ocr,
        _ => VlmTask::Ocr,
    }
}

/// Whether the pipeline should run VLM on this layout label (gated by profile flags).
///
/// Matches PaddleX `paddleocr_vl/pipeline.py`: image-like labels skip VLM unless
/// enabled; `markdown_ignore_labels` only affects Markdown assembly, not recognition.
pub fn should_run_vlm_for_label(
    label: &str,
    use_chart_recognition: bool,
    use_seal_recognition: bool,
    use_ocr_for_image_block: bool,
) -> bool {
    match label {
        "chart" => use_chart_recognition,
        "seal" => use_seal_recognition,
        "image" | "header_image" | "footer_image" => use_ocr_for_image_block,
        _ => true,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmConfig {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub torch_dtype: String,
}

impl VlmConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        #[derive(Deserialize)]
        struct Root {
            hidden_size: u32,
            vocab_size: u32,
            torch_dtype: String,
        }
        let root: Root = serde_json::from_str(&data)?;
        Ok(Self {
            hidden_size: root.hidden_size,
            vocab_size: root.vocab_size,
            torch_dtype: root.torch_dtype,
        })
    }
}

pub struct VlmModel {
    model_dir: PathBuf,
    config: VlmConfig,
    device: Device,
    runner: std::sync::Mutex<Option<model::VlmRunner>>,
}

impl VlmModel {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(VlmError::Message(format!(
                "missing model weights at {}",
                weights.display()
            )));
        }
        let config = VlmConfig::from_dir(&model_dir)?;
        Ok(Self {
            model_dir,
            config,
            device,
            runner: std::sync::Mutex::new(None),
        })
    }

    fn runner(&self) -> Result<std::sync::MutexGuard<'_, Option<model::VlmRunner>>> {
        let mut guard = self
            .runner
            .lock()
            .map_err(|_| VlmError::LockPoisoned)?;
        if guard.is_none() {
            *guard = Some(model::VlmRunner::load(&self.model_dir, self.device.clone())?);
        }
        Ok(guard)
    }

    fn runner_mut<'a>(
        guard: &'a mut std::sync::MutexGuard<'_, Option<model::VlmRunner>>,
    ) -> Result<&'a mut model::VlmRunner> {
        guard
            .as_mut()
            .ok_or(VlmError::RunnerNotLoaded)
    }

    fn runner_ref<'a>(
        guard: &'a std::sync::MutexGuard<'_, Option<model::VlmRunner>>,
    ) -> Result<&'a model::VlmRunner> {
        guard
            .as_ref()
            .ok_or(VlmError::RunnerNotLoaded)
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn config(&self) -> &VlmConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn generate(
        &self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<String> {
        let mut guard = self.runner()?;
        Self::runner_mut(&mut guard)?
            .generate(image, task, max_new_tokens)
            .map_err(Into::into)
    }

    /// Greedy-decode token ids (parity vs HF goldens).
    pub fn generate_token_ids(
        &self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut guard = self.runner()?;
        Self::runner_mut(&mut guard)?
            .generate_token_ids(image, task, max_new_tokens)
            .map_err(Into::into)
    }

    pub fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let guard = self.runner()?;
        Self::runner_ref(&guard)?.decode_token_ids(tokens).map_err(Into::into)
    }

    pub fn preprocess_grid_thw(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<Vec<u32>>> {
        let guard = self.runner()?;
        Self::runner_ref(&guard)?
            .preprocess_grid_thw(image, task)
            .map_err(Into::into)
    }

    pub fn generate_dynamic(
        &self,
        image: &DynamicImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<String> {
        self.generate(&image.to_rgb8(), task, max_new_tokens)
    }

    pub fn generate_from_path(
        &self,
        path: impl AsRef<Path>,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<String> {
        let rgb = image::open(path.as_ref())
            .with_context(|| format!("open image {}", path.as_ref().display()))?
            .to_rgb8();
        self.generate(&rgb, task, max_new_tokens)
    }

    /// Build input token ids for parity tests (without running the model).
    pub fn preprocess_input_ids_len(
        &self,
        image: &RgbImage,
        task: VlmTask,
    ) -> Result<usize> {
        Ok(self.preprocess_input_ids(image, task)?.len())
    }

    pub fn preprocess_input_ids(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<u32>> {
        let guard = self.runner()?;
        Self::runner_ref(&guard)?
            .build_input_ids_vec(image, task)
            .map_err(Into::into)
    }
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    Ok(docparser_candle_utils::list_safetensor_keys(model_dir)?)
}
