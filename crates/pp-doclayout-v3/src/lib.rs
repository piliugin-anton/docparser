use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use serde::{Deserialize, Serialize};

mod model;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutElement {
    pub id: usize,
    pub order: Option<usize>,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    pub num_labels: u32,
    pub num_queries: u32,
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
        Ok(Self {
            num_labels: root.id2label.len() as u32,
            num_queries: root.num_queries,
        })
    }
}

pub struct LayoutModel {
    model_dir: PathBuf,
    config: LayoutConfig,
}

impl LayoutModel {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            bail!("missing layout weights at {}", weights.display());
        }
        let config = LayoutConfig::from_dir(&model_dir)?;
        Ok(Self { model_dir, config })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn config(&self) -> &LayoutConfig {
        &self.config
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        model::detect(&self.model_dir, image)
    }

    pub fn detect_path(&self, path: impl AsRef<Path>) -> Result<Vec<LayoutElement>> {
        let rgb = image::open(path.as_ref())
            .with_context(|| format!("open image {}", path.as_ref().display()))?
            .to_rgb8();
        self.detect(&rgb)
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
        9 => "footer",
        10 => "footnote",
        11 => "formula_number",
        12 => "header",
        13 => "header",
        14 => "image",
        15 => "formula",
        16 => "number",
        17 => "paragraph_title",
        18 => "reference",
        19 => "reference_content",
        20 => "seal",
        21 => "table",
        22 => "text",
        23 => "text",
        24 => "vision_footnote",
        _ => "unknown",
    }
}
