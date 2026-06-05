#![deny(unsafe_code)]

use std::path::Path;

mod error;

use candle_core::Device;
use docparser_candle_utils::LazyRunner;
pub use error::{Result, VlmError};
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
        let data = docparser_candle_utils::read_json_from_dir(model_dir, "config.json")?;
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
    config: VlmConfig,
    device: Device,
    runner: LazyRunner<model::VlmRunner>,
}

impl VlmModel {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = VlmConfig::from_dir(&model_dir)?;
        Ok(Self {
            config,
            device,
            runner: LazyRunner::new(model_dir),
        })
    }

    pub fn model_dir(&self) -> &Path {
        self.runner.model_dir()
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
        let device = self.device.clone();
        self.runner.with_runner_mut(
            move |dir| model::VlmRunner::load(dir, device),
            |r| r.generate(image, task, max_new_tokens),
        )
    }

    /// Greedy-decode token ids (parity vs HF goldens).
    pub fn generate_token_ids(
        &self,
        image: &RgbImage,
        task: VlmTask,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let device = self.device.clone();
        self.runner.with_runner_mut(
            move |dir| model::VlmRunner::load(dir, device),
            |r| r.generate_token_ids(image, task, max_new_tokens),
        )
    }

    pub fn decode_token_ids(&self, tokens: &[u32]) -> Result<String> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| model::VlmRunner::load(dir, device),
            |r| r.decode_token_ids(tokens),
        )
    }

    pub fn preprocess_grid_thw(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<Vec<u32>>> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| model::VlmRunner::load(dir, device),
            |r| r.preprocess_grid_thw(image, task),
        )
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
            .map_err(VlmError::Image)?
            .to_rgb8();
        self.generate(&rgb, task, max_new_tokens)
    }

    /// Build input token ids for parity tests (without running the model).
    pub fn preprocess_input_ids_len(&self, image: &RgbImage, task: VlmTask) -> Result<usize> {
        Ok(self.preprocess_input_ids(image, task)?.len())
    }

    pub fn preprocess_input_ids(&self, image: &RgbImage, task: VlmTask) -> Result<Vec<u32>> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| model::VlmRunner::load(dir, device),
            |r| r.build_input_ids_vec(image, task),
        )
    }
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    Ok(docparser_candle_utils::list_safetensor_keys(model_dir)?)
}
