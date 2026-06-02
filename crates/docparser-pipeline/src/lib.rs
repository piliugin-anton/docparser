use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::Device;
use image::{DynamicImage, GenericImageView, RgbImage};
use paddleocr_vl::{VlmModel, task_for_layout_label};
use pp_doclayout_v3::LayoutModel;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub max_tokens: usize,
    pub unclip_ratio: f32,
    pub include_markdown: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            unclip_ratio: 0.02,
            include_markdown: true,
        }
    }
}

pub struct DocumentPipeline {
    layout: LayoutModel,
    vlm: VlmModel,
    config: PipelineConfig,
    vl_model_name: String,
    layout_model_name: String,
}

impl DocumentPipeline {
    pub fn from_dirs(
        vlm_dir: impl AsRef<Path>,
        layout_dir: impl AsRef<Path>,
        config: PipelineConfig,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let vlm_dir = vlm_dir.as_ref().to_path_buf();
        let layout_dir = layout_dir.as_ref().to_path_buf();

        let vlm = VlmModel::from_dir(&vlm_dir, device)?;
        let layout = LayoutModel::from_dir(&layout_dir)?;

        Ok(Self {
            vl_model_name: vlm_dir.display().to_string(),
            layout_model_name: layout_dir.display().to_string(),
            layout,
            vlm,
            config,
        })
    }

    pub fn parse_image(
        &self,
        image: DynamicImage,
        filename: Option<String>,
    ) -> Result<DocumentParseResult> {
        let started = std::time::Instant::now();
        let (width, height) = image.dimensions();
        let rgb = image.to_rgb8();

        let mut layout_elements = self.layout.detect(&rgb)?;
        if layout_elements.is_empty() {
            // Small or atypical pages may yield no boxes above the layout threshold; run VLM on the full image.
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
            let crop = crop_bbox(&rgb, el.bbox, self.config.unclip_ratio);
            let task = task_for_layout_label(&el.label);
            let content = self.vlm.generate(&crop, task, self.config.max_tokens)?;
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
            Some(blocks_to_markdown(&blocks))
        } else {
            None
        };

        Ok(DocumentParseResult {
            document_id: Uuid::new_v4(),
            source: SourceInfo {
                filename,
                width,
                height,
                format: "rgb".into(),
            },
            pipeline_version: "v1".into(),
            stages: vec![
                "pp_doclayout_v3".into(),
                "paddleocr_vl_1.6".into(),
            ],
            blocks,
            markdown,
            metadata: ParseMetadata {
                vl_model: self.vl_model_name.clone(),
                layout_model: self.layout_model_name.clone(),
                processing_ms: started.elapsed().as_millis() as u64,
                device: "cpu".into(),
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

fn blocks_to_markdown(blocks: &[Block]) -> String {
    const SKIP: &[&str] = &[
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
        "formula_number",
    ];

    let mut out = String::new();
    for block in blocks {
        if SKIP.contains(&block.label.as_str()) {
            continue;
        }
        if !block.content.is_empty() {
            out.push_str(&block.content);
            if !block.content.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
    }
    out
}

pub fn default_model_paths(base: impl AsRef<Path>) -> (PathBuf, PathBuf) {
    let base = base.as_ref();
    (
        base.join("PaddleOCR-VL-1.6"),
        base.join("PP-DocLayoutV3"),
    )
}
