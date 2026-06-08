#![deny(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod error;

use candle_core::Device;
use docparser_candle_utils::{LazyRunner, device_from_env};
pub use error::{LayoutError, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};

mod image_processor;
mod model;
mod postprocess;
pub mod pp_doclayout_v3;
mod preprocess;

pub use image_processor::{LayoutImageProcessor, LayoutPreprocessorConfig};
pub use pp_doclayout_v3::PpDocLayoutV3Config;
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

/// Layout model metadata loaded from `config.json` (via [`PpDocLayoutV3Config`]).
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    inner: PpDocLayoutV3Config,
    pub detection_threshold: f32,
}

impl LayoutConfig {
    pub fn from_dir(model_dir: &Path, detection_threshold: f32) -> Result<Self> {
        let inner = PpDocLayoutV3Config::from_dir(model_dir)?;
        Ok(Self {
            inner,
            detection_threshold,
        })
    }

    pub fn num_labels(&self) -> u32 {
        self.inner.num_labels() as u32
    }

    pub fn num_queries(&self) -> u32 {
        self.inner.num_queries as u32
    }

    pub fn id2label(&self) -> HashMap<u32, String> {
        self.inner.id2label_map()
    }

    pub fn label_for_id(&self, id: i64) -> String {
        self.inner.label_for_id(id)
    }
}

pub struct LayoutModel {
    model_dir: PathBuf,
    config: LayoutConfig,
    device: Device,
    runner: LazyRunner<model::LayoutRunner>,
}

impl LayoutModel {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let device = device_from_env()?;
        Self::from_dir_with_threshold(model_dir, 0.5, device)
    }

    pub fn from_dir_with_threshold(
        model_dir: impl AsRef<Path>,
        detection_threshold: f32,
        device: Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = LayoutConfig::from_dir(&model_dir, detection_threshold)?;
        Ok(Self {
            model_dir: model_dir.clone(),
            config,
            device,
            runner: LazyRunner::new(model_dir),
        })
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
            return model::LayoutRunner::load(&self.model_dir, t, self.device.clone())?
                .detect(image);
        }
        let device = self.device.clone();
        let threshold = self.config.detection_threshold;
        self.runner.with_runner(
            move |dir| model::LayoutRunner::load(dir, threshold, device.clone()),
            |r| r.detect(image),
        )
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
    Ok(PpDocLayoutV3Config::from_dir(model_dir)?.label_for_id(id))
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
