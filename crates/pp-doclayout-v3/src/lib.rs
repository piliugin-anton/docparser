#![deny(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod error;

pub use error::{LayoutError, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};

mod image_processor;
mod model;
mod postprocess;
pub mod pp_doclayout_v3;
mod preprocess;

pub use image_processor::{LayoutImageProcessor, LayoutPreprocessorConfig};
pub use preprocess::PreprocessOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutElement {
    pub id: usize,
    pub order: Option<usize>,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LayoutConfig {
    pub num_labels: u32,
    pub num_queries: u32,
    pub id2label: HashMap<u32, String>,
    pub detection_threshold: f32,
}

impl LayoutConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&path)?;
        #[derive(Deserialize)]
        struct Root {
            num_queries: u32,
            id2label: serde_json::Map<String, serde_json::Value>,
        }
        let root: Root = serde_json::from_str(&data)?;
        let id2label: HashMap<u32, String> = root
            .id2label
            .iter()
            .filter_map(|(k, v)| {
                let id: u32 = k.parse().ok()?;
                let name = v.as_str()?.to_string();
                Some((id, name))
            })
            .collect();
        Ok(Self {
            num_labels: id2label.len() as u32,
            num_queries: root.num_queries,
            id2label,
            detection_threshold: 0.5,
        })
    }

    pub fn label_for_id(&self, id: i64) -> String {
        self.id2label
            .get(&(id as u32))
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

pub struct LayoutModel {
    model_dir: PathBuf,
    config: LayoutConfig,
    runner: std::sync::Mutex<Option<model::LayoutRunner>>,
}

impl LayoutModel {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with_threshold(model_dir, 0.5)
    }

    pub fn from_dir_with_threshold(
        model_dir: impl AsRef<Path>,
        detection_threshold: f32,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(LayoutError::Message(format!(
                "missing layout weights at {}",
                weights.display()
            )));
        }
        let mut config = LayoutConfig::from_dir(&model_dir)?;
        config.detection_threshold = detection_threshold;
        Ok(Self {
            model_dir,
            config,
            runner: std::sync::Mutex::new(None),
        })
    }

    fn runner(&self) -> Result<std::sync::MutexGuard<'_, Option<model::LayoutRunner>>> {
        let mut guard = self.runner.lock().map_err(|_| LayoutError::LockPoisoned)?;
        if guard.is_none() {
            *guard = Some(model::LayoutRunner::load(
                &self.model_dir,
                self.config.detection_threshold,
            )?);
        }
        Ok(guard)
    }

    fn runner_ref<'a>(
        guard: &'a std::sync::MutexGuard<'_, Option<model::LayoutRunner>>,
    ) -> Result<&'a model::LayoutRunner> {
        guard.as_ref().ok_or(LayoutError::RunnerNotLoaded)
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn config(&self) -> &LayoutConfig {
        &self.config
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        self.detect_with_options(image, None)
    }

    pub fn detect_with_options(
        &self,
        image: &RgbImage,
        threshold: Option<f32>,
    ) -> Result<Vec<LayoutElement>> {
        let t = threshold.unwrap_or(self.config.detection_threshold);
        if (t - self.config.detection_threshold).abs() > f32::EPSILON {
            return model::LayoutRunner::load(&self.model_dir, t)?.detect(image);
        }
        let guard = self.runner()?;
        Self::runner_ref(&guard)?.detect(image)
    }

    pub fn detect_path(&self, path: impl AsRef<Path>) -> Result<Vec<LayoutElement>> {
        let rgb = image::open(path.as_ref())
            .map_err(LayoutError::Image)?
            .to_rgb8();
        self.detect(&rgb)
    }
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    Ok(docparser_candle_utils::list_safetensor_keys(model_dir)?)
}

/// Resolve label name from id using a model directory's `config.json`.
pub fn label_name_from_dir(model_dir: &Path, id: i64) -> Result<String> {
    Ok(LayoutConfig::from_dir(model_dir)?.label_for_id(id))
}

#[deprecated(note = "use LayoutConfig::label_for_id or label_name_from_dir")]
pub fn label_name(id: i64) -> &'static str {
    match id {
        0 => "abstract",
        1 => "algorithm",
        2 => "aside_text",
        3 => "chart",
        4 => "content",
        5 => "formula",
        6 => "doc_title",
        7 => "figure_title",
        8 => "footer",
        9 => "footer_image",
        10 => "footnote",
        11 => "formula_number",
        12 => "header",
        13 => "header_image",
        14 => "image",
        15 => "inline_formula",
        16 => "number",
        17 => "paragraph_title",
        18 => "reference",
        19 => "reference_content",
        20 => "seal",
        21 => "table",
        22 => "text",
        23 => "text_block",
        24 => "vision_footnote",
        _ => "unknown",
    }
}
