use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::Device;
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};

mod model;

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

pub fn task_for_layout_label(label: &str) -> VlmTask {
    match label {
        "table" => VlmTask::Table,
        "display_formula" | "inline_formula" | "formula" => VlmTask::Formula,
        "chart" => VlmTask::Chart,
        "seal" => VlmTask::Seal,
        _ => VlmTask::Ocr,
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
}

impl VlmModel {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            bail!("missing model weights at {}", weights.display());
        }
        let config = VlmConfig::from_dir(&model_dir)?;
        Ok(Self {
            model_dir,
            config,
            device,
        })
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
        model::generate(&self.model_dir, &self.device, image, task, max_new_tokens)
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
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    let path = model_dir.join("model.safetensors");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let data = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut keys: Vec<String> = data.names().into_iter().cloned().collect();
    keys.sort();
    Ok(keys)
}
