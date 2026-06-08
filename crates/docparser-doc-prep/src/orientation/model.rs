use std::path::Path;

use candle_core::{DType, Device};
use docparser_candle_utils::LazyRunner;
use image::{DynamicImage, RgbImage};

use super::config::PpLcnetConfig;
use super::nn::PpLcnetModel;
use super::preprocess::{PreprocessorConfig, preprocess};
use super::rotate::rotate_by_angle;
use super::{DocOriError, Result};

pub struct DocOrientationRunner {
    model: PpLcnetModel,
    config: PpLcnetConfig,
    preprocessor: PreprocessorConfig,
    device: Device,
}

impl DocOrientationRunner {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let config = PpLcnetConfig::from_dir(model_dir)?;
        let preprocessor = PreprocessorConfig::from_dir(model_dir)?;
        let vb =
            docparser_candle_utils::var_builder_from_safetensors(model_dir, DType::F32, &device)?;
        let model = PpLcnetModel::load(&config, vb)?;
        Ok(Self {
            model,
            config,
            preprocessor,
            device,
        })
    }

    pub fn classify(&self, image: &RgbImage) -> Result<(u32, f32)> {
        let pixel_values = preprocess(image, &self.preprocessor, &self.device)?;
        let probs = self.model.forward(&pixel_values)?;
        let probs = probs.squeeze(0)?.to_vec1::<f32>()?;
        let (class_id, score) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i, s))
            .unwrap_or((0, 0.0));
        let angle = self.config.angle_for_class(class_id);
        Ok((angle, score))
    }

    pub fn logits(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let pixel_values = preprocess(image, &self.preprocessor, &self.device)?;
        self.model
            .forward_logits(&pixel_values)?
            .squeeze(0)?
            .to_vec1::<f32>()
            .map_err(DocOriError::Candle)
    }

    pub fn predict_and_rotate(&self, image: DynamicImage) -> Result<(DynamicImage, u32)> {
        let rgb = image.to_rgb8();
        let (angle, _score) = self.classify(&rgb)?;
        Ok((rotate_by_angle(image, angle), angle))
    }
}

pub struct DocOrientationModel {
    device: Device,
    runner: LazyRunner<DocOrientationRunner>,
}

impl DocOrientationModel {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        Ok(Self {
            device,
            runner: LazyRunner::new("doc_orientation", model_dir.as_ref().to_path_buf()),
        })
    }

    pub fn classify(&self, image: &RgbImage) -> Result<(u32, f32)> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| DocOrientationRunner::load(dir, device.clone()),
            |r| r.classify(image),
        )
    }

    pub fn logits(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| DocOrientationRunner::load(dir, device.clone()),
            |r| r.logits(image),
        )
    }

    pub fn predict_and_rotate(&self, image: DynamicImage) -> Result<(DynamicImage, u32)> {
        let device = self.device.clone();
        self.runner.with_runner(
            move |dir| DocOrientationRunner::load(dir, device.clone()),
            |r| r.predict_and_rotate(image),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn softmax_argmax() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![0.1f32, 0.7, 0.15, 0.05], (1, 4), &device).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&t).unwrap();
        let v = probs.to_vec2::<f32>().unwrap();
        assert!(v[0][1] > v[0][0]);
    }
}
