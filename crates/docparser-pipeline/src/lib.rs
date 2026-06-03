mod doc_preprocess;
mod layout_merge;
mod layout_nms;
mod markdown;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::Device;
use image::{DynamicImage, GenericImageView, RgbImage};
use paddleocr_vl::{should_run_vlm_for_label, task_for_layout_label, VlmModel};
use pp_doclayout_v3::LayoutModel;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use doc_preprocess::{preprocess_document, DocPreprocessor, DocPreprocessorConfig};
pub use layout_merge::{
    merge_layout_blocks, merge_layout_blocks_with_mode_fn, merge_mode_for_label, MergeBboxesMode,
};
pub use layout_nms::layout_nms;
pub use markdown::{blocks_to_markdown, official_markdown_ignore_labels};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub filename: Option<String>,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: usize,
    pub order: Option<usize>,
    pub label: String,
    pub bbox: [f32; 4],
    pub score: f32,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseMetadata {
    pub vl_model: String,
    pub layout_model: String,
    pub processing_ms: u64,
    pub device: String,
    pub pipeline_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentParseResult {
    pub document_id: Uuid,
    pub source: SourceInfo,
    pub pipeline_version: String,
    pub stages: Vec<String>,
    pub blocks: Vec<Block>,
    pub markdown: Option<String>,
    pub metadata: ParseMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub max_tokens: usize,
    pub layout_threshold: f32,
    pub layout_unclip_ratio: f32,
    pub layout_nms: bool,
    pub layout_nms_iou: f32,
    pub merge_layout_blocks: bool,
    pub markdown_ignore_labels: Vec<String>,
    pub use_chart_recognition: bool,
    pub use_seal_recognition: bool,
    pub use_ocr_for_image_block: bool,
    pub include_markdown: bool,
    pub doc_preprocess: DocPreprocessorConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            layout_threshold: 0.3,
            layout_unclip_ratio: 1.0,
            layout_nms: true,
            layout_nms_iou: 0.5,
            merge_layout_blocks: true,
            markdown_ignore_labels: official_markdown_ignore_labels(),
            use_chart_recognition: false,
            use_seal_recognition: false,
            use_ocr_for_image_block: false,
            include_markdown: true,
            doc_preprocess: DocPreprocessorConfig::default(),
        }
    }
}

impl PipelineConfig {
    pub const PIPELINE_VERSION: &'static str = "v1.6";
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub vlm: PathBuf,
    pub layout: PathBuf,
    pub doc_ori: PathBuf,
    pub uvdoc: PathBuf,
}

impl ModelPaths {
    pub fn from_models_dir(base: impl AsRef<Path>) -> Self {
        let base = base.as_ref();
        Self {
            vlm: base.join("PaddleOCR-VL-1.6"),
            layout: base.join("PP-DocLayoutV3"),
            doc_ori: base.join("PP-LCNet_x1_0_doc_ori"),
            uvdoc: base.join("UVDoc"),
        }
    }
}

pub struct DocumentPipeline {
    layout: LayoutModel,
    vlm: VlmModel,
    doc_prep: DocPreprocessor,
    config: PipelineConfig,
    vl_model_name: String,
    layout_model_name: String,
}

impl DocumentPipeline {
    pub fn from_models_dir(models_dir: impl AsRef<Path>, config: PipelineConfig) -> Result<Self> {
        let paths = ModelPaths::from_models_dir(models_dir);
        Self::from_paths(&paths, config)
    }

    pub fn from_paths(paths: &ModelPaths, config: PipelineConfig) -> Result<Self> {
        let device = Device::Cpu;
        let vlm = VlmModel::from_dir(&paths.vlm, device)?;
        let layout =
            LayoutModel::from_dir_with_threshold(&paths.layout, config.layout_threshold)?;
        let doc_prep = DocPreprocessor::from_model_dirs(
            Some(&paths.doc_ori),
            Some(&paths.uvdoc),
            &config.doc_preprocess,
        )?;

        Ok(Self {
            vl_model_name: paths.vlm.display().to_string(),
            layout_model_name: paths.layout.display().to_string(),
            layout,
            vlm,
            doc_prep,
            config,
        })
    }

    /// Load VLM + layout from explicit dirs (doc prep models resolved from sibling paths when present).
    pub fn from_dirs(
        vlm_dir: impl AsRef<Path>,
        layout_dir: impl AsRef<Path>,
        config: PipelineConfig,
    ) -> Result<Self> {
        let vlm_dir = vlm_dir.as_ref();
        let layout_dir = layout_dir.as_ref();
        let base = vlm_dir
            .parent()
            .unwrap_or_else(|| Path::new("models"));
        let mut paths = ModelPaths::from_models_dir(base);
        paths.vlm = vlm_dir.to_path_buf();
        paths.layout = layout_dir.to_path_buf();
        Self::from_paths(&paths, config)
    }

    pub fn parse_image(
        &self,
        image: DynamicImage,
        filename: Option<String>,
    ) -> Result<DocumentParseResult> {
        let started = std::time::Instant::now();
        let (image, prep_stages) = self
            .doc_prep
            .preprocess_document(image, &self.config.doc_preprocess)?;
        let (width, height) = image.dimensions();
        let rgb = image.to_rgb8();

        let mut layout_elements = self.layout.detect(&rgb)?;
        if self.config.layout_nms {
            layout_elements = layout_nms(layout_elements, self.config.layout_nms_iou);
        }
        if self.config.merge_layout_blocks {
            layout_elements =
                merge_layout_blocks_with_mode_fn(layout_elements, merge_mode_for_label);
        }

        if layout_elements.is_empty() {
            tracing::debug!("no layout boxes above threshold; using full-image VLM fallback");
            layout_elements.push(pp_doclayout_v3::LayoutElement {
                id: 0,
                order: Some(0),
                label: "text".into(),
                score: 1.0,
                bbox: [0.0, 0.0, width as f32, height as f32],
                text: None,
            });
        }
        layout_elements.sort_by(|a, b| {
            a.order
                .unwrap_or(usize::MAX)
                .cmp(&b.order.unwrap_or(usize::MAX))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut blocks = Vec::new();
        for el in &layout_elements {
            let content = if should_run_vlm_for_label(
                &el.label,
                self.config.use_chart_recognition,
                self.config.use_seal_recognition,
                self.config.use_ocr_for_image_block,
            ) {
                let crop = crop_bbox(&rgb, el.bbox, self.config.layout_unclip_ratio);
                let task = task_for_layout_label(&el.label);
                self.vlm.generate(&crop, task, self.config.max_tokens)?
            } else {
                String::new()
            };
            blocks.push(Block {
                id: el.id,
                order: el.order,
                label: el.label.clone(),
                bbox: el.bbox,
                score: el.score,
                content,
            });
        }

        let markdown = if self.config.include_markdown {
            Some(blocks_to_markdown(
                &blocks,
                &self.config.markdown_ignore_labels,
            ))
        } else {
            None
        };

        let mut stages: Vec<String> = prep_stages.into_iter().map(str::to_string).collect();
        stages.push("pp_doclayout_v3".into());
        stages.push("paddleocr_vl_1.6".into());

        Ok(DocumentParseResult {
            document_id: Uuid::new_v4(),
            source: SourceInfo {
                filename,
                width,
                height,
                format: "rgb".into(),
            },
            pipeline_version: PipelineConfig::PIPELINE_VERSION.into(),
            stages,
            blocks,
            markdown,
            metadata: ParseMetadata {
                vl_model: self.vl_model_name.clone(),
                layout_model: self.layout_model_name.clone(),
                processing_ms: started.elapsed().as_millis() as u64,
                device: "cpu".into(),
                pipeline_version: PipelineConfig::PIPELINE_VERSION.into(),
            },
        })
    }

    pub fn parse_path(&self, path: impl AsRef<Path>) -> Result<DocumentParseResult> {
        let path = path.as_ref();
        let filename = path.file_name().map(|s| s.to_string_lossy().into_owned());
        let img = image::open(path).with_context(|| format!("open image {}", path.display()))?;
        self.parse_image(img, filename)
    }
}

fn crop_bbox(image: &RgbImage, bbox: [f32; 4], unclip_ratio: f32) -> RgbImage {
    let (w, h) = image.dimensions();
    let [x1, y1, x2, y2] = bbox;
    let bw = (x2 - x1).max(1.0);
    let bh = (y2 - y1).max(1.0);
    let pad_x = bw * unclip_ratio;
    let pad_y = bh * unclip_ratio;
    let x1 = (x1 - pad_x).max(0.0).floor() as u32;
    let y1 = (y1 - pad_y).max(0.0).floor() as u32;
    let x2 = (x2 + pad_x).min(w as f32).ceil() as u32;
    let y2 = (y2 + pad_y).min(h as f32).ceil() as u32;
    image::imageops::crop_imm(
        image,
        x1,
        y1,
        x2.saturating_sub(x1).max(1),
        y2.saturating_sub(y1).max(1),
    )
    .to_image()
}

pub fn default_model_paths(base: impl AsRef<Path>) -> (PathBuf, PathBuf) {
    let paths = ModelPaths::from_models_dir(base);
    (paths.vlm, paths.layout)
}

pub fn model_paths(base: impl AsRef<Path>) -> ModelPaths {
    ModelPaths::from_models_dir(base)
}
