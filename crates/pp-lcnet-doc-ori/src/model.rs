use std::path::Path;
use std::sync::Mutex;

use crate::{DocOriError, Result};
use anyhow::{Result as AnyhowResult};
use candle_core::{Device, DType};
use image::{DynamicImage, RgbImage};

use crate::config::PpLcnetConfig;
use crate::nn::PpLcnetModel;
use crate::preprocess::{preprocess, PreprocessorConfig};
use crate::rotate::rotate_by_angle;

pub struct DocOrientationRunner {
    model: PpLcnetModel,
    config: PpLcnetConfig,
    preprocessor: PreprocessorConfig,
    device: Device,
}

impl DocOrientationRunner {
    pub fn load(model_dir: &Path) -> AnyhowResult<Self> {
        let device = Device::Cpu;
        let config = PpLcnetConfig::from_dir(model_dir)?;
        let preprocessor = PreprocessorConfig::from_dir(model_dir)?;
        let vb = docparser_candle_utils::var_builder_from_safetensors(model_dir, DType::F32, &device)
            .map_err(anyhow::Error::from)?;
        let model = PpLcnetModel::load(&config, vb)?;
        Ok(Self {
            model,
            config,
            preprocessor,
            device,
        })
    }

    pub fn classify(&self, image: &RgbImage) -> AnyhowResult<(u32, f32)> {
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

    pub fn logits(&self, image: &RgbImage) -> AnyhowResult<Vec<f32>> {
        let pixel_values = preprocess(image, &self.preprocessor, &self.device)?;
        self.model
            .forward_logits(&pixel_values)?
            .squeeze(0)?
            .to_vec1::<f32>()
            .map_err(Into::into)
    }

    pub fn predict_and_rotate(&self, image: DynamicImage) -> AnyhowResult<(DynamicImage, u32)> {
        let rgb = image.to_rgb8();
        let (angle, _score) = self.classify(&rgb)?;
        Ok((rotate_by_angle(image, angle), angle))
    }
}

pub struct DocOrientationModel {
    model_dir: std::path::PathBuf,
    runner: Mutex<Option<DocOrientationRunner>>,
}

impl DocOrientationModel {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(DocOriError::Message(format!(
                "missing weights at {}",
                weights.display()
            )));
        }
        Ok(Self {
            model_dir,
            runner: Mutex::new(None),
        })
    }

    fn runner(&self) -> Result<std::sync::MutexGuard<'_, Option<DocOrientationRunner>>> {
        let mut guard = self
            .runner
            .lock()
            .map_err(|_| DocOriError::LockPoisoned)?;
        if guard.is_none() {
            *guard = Some(DocOrientationRunner::load(&self.model_dir)?);
        }
        Ok(guard)
    }

    fn runner_ref<'a>(
        guard: &'a std::sync::MutexGuard<'_, Option<DocOrientationRunner>>,
    ) -> Result<&'a DocOrientationRunner> {
        guard
            .as_ref()
            .ok_or(DocOriError::RunnerNotLoaded)
    }

    pub fn classify(&self, image: &RgbImage) -> Result<(u32, f32)> {
        Self::runner_ref(&self.runner()?)?.classify(image).map_err(Into::into)
    }

    pub fn logits(&self, image: &RgbImage) -> Result<Vec<f32>> {
        Self::runner_ref(&self.runner()?)?.logits(image).map_err(Into::into)
    }

    pub fn predict_and_rotate(&self, image: DynamicImage) -> Result<(DynamicImage, u32)> {
        Self::runner_ref(&self.runner()?)?
            .predict_and_rotate(image)
            .map_err(Into::into)
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
