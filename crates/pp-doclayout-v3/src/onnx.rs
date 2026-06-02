//! ONNX Runtime inference for PP-DocLayoutV3.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use image::RgbImage;
use ort::session::Session;
use ort::value::TensorRef;

use crate::postprocess::decode_detections;
use crate::preprocess::preprocess;
use crate::LayoutElement;

pub struct OnnxLayoutModel {
    session: Arc<Mutex<Session>>,
}

impl OnnxLayoutModel {
    pub fn load(onnx_path: &Path) -> Result<Self> {
        let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("load onnx {}", onnx_path.display()))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
        })
    }

    pub fn detect(&self, image: &RgbImage) -> Result<Vec<LayoutElement>> {
        let prep = preprocess(image)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("onnx session lock poisoned"))?;

        let im_shape = TensorRef::from_array_view(prep.im_shape.view())?;
        let image_t = TensorRef::from_array_view(prep.pixel_values.view())?;
        let scale = TensorRef::from_array_view(prep.scale_factor.view())?;

        let mut outputs = session.run(ort::inputs![
            "im_shape" => im_shape,
            "image" => image_t,
            "scale_factor" => scale,
        ])?;

        let dets = outputs
            .remove("fetch_name_0")
            .or_else(|| outputs.into_iter().next().map(|(_, v)| v))
            .context("onnx missing detection output")?;
        let (_shape, flat) = dets.try_extract_tensor::<f32>()?;
        let n = flat.len() / 7;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 7;
            rows.push([
                flat[base],
                flat[base + 1],
                flat[base + 2],
                flat[base + 3],
                flat[base + 4],
                flat[base + 5],
                flat[base + 6],
            ]);
        }
        Ok(decode_detections(&rows))
    }
}
