//! Layout detection entry point (ONNX graph + Candle ecosystem weights layout).

use std::path::Path;

use anyhow::Result;
use image::RgbImage;

use crate::onnx::OnnxLayoutModel;
use crate::LayoutElement;
use docparser_candle_utils::default_onnx_layout_path;

pub struct LayoutRunner {
    onnx: OnnxLayoutModel,
}

impl LayoutRunner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let onnx_path = default_onnx_layout_path(model_dir);
        anyhow::ensure!(
            onnx_path.is_file(),
            "missing layout ONNX at {} (run docparser-download with layout ONNX)",
            onnx_path.display()
        );
        let onnx = OnnxLayoutModel::load(&onnx_path)?;
        Ok(Self { onnx })
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        self.onnx.detect(image)
    }
}

pub fn detect(model_dir: &Path, image: &RgbImage) -> Result<Vec<LayoutElement>> {
    let runner = LayoutRunner::load(model_dir)?;
    runner.detect(image)
}
