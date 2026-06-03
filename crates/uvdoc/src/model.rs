use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Result};
use candle_core::{Device, DType, Tensor};
use image::RgbImage;

use crate::config::UvdocConfig;
use crate::nn::UvdocNet;
use crate::preprocess::{preprocess_with_original, PreprocessorConfig};

pub struct UvdocRunner {
    model: UvdocNet,
    preprocessor: PreprocessorConfig,
    device: Device,
}

impl UvdocRunner {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let config = UvdocConfig::from_dir(model_dir)?;
        let preprocessor = PreprocessorConfig::from_dir(model_dir)?;
        let vb = docparser_candle_utils::var_builder_from_safetensors(model_dir, DType::F32, &device)?;
        let model = UvdocNet::load(&config, vb)?;
        Ok(Self {
            model,
            preprocessor,
            device,
        })
    }

    pub fn rectify(&self, image: &RgbImage) -> Result<RgbImage> {
        let prep = preprocess_with_original(image, &self.preprocessor, &self.device)?;
        let flow = self.model.forward_flow(&prep.network_input)?;
        let output = self
            .model
            .rectify_with_flow(&prep.original_bgr, &flow)?;
        tensor_bgr_to_rgb(&output)
    }

    pub fn forward_flow(&self, image: &RgbImage) -> Result<Tensor> {
        let prep = preprocess_with_original(image, &self.preprocessor, &self.device)?;
        Ok(self.model.forward_flow(&prep.network_input)?)
    }
}

fn tensor_bgr_to_rgb(out: &Tensor) -> Result<RgbImage> {
    let out = out.squeeze(0)?.to_dtype(DType::F32)?;
    let (c, h, w) = out.dims3()?;
    if c != 3 {
        bail!("expected 3-channel output, got {c}");
    }
    let data = out.flatten_all()?.to_vec1::<f32>()?;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let b = (data[0 * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data[1 * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let r = (data[2 * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    Ok(img)
}

pub struct UvdocModel {
    model_dir: std::path::PathBuf,
    runner: Mutex<Option<UvdocRunner>>,
}

impl UvdocModel {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let weights = model_dir.join("model.safetensors");
        if !weights.is_file() {
            bail!("missing weights at {}", weights.display());
        }
        Ok(Self {
            model_dir,
            runner: Mutex::new(None),
        })
    }

    fn runner(&self) -> Result<std::sync::MutexGuard<'_, Option<UvdocRunner>>> {
        let mut guard = self
            .runner
            .lock()
            .map_err(|_| anyhow::anyhow!("uvdoc runner lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(UvdocRunner::load(&self.model_dir)?);
        }
        Ok(guard)
    }

    pub fn rectify(&self, image: &RgbImage) -> Result<RgbImage> {
        self.runner()?.as_ref().unwrap().rectify(image)
    }

    pub fn forward_flow(&self, image: &RgbImage) -> Result<candle_core::Tensor> {
        self.runner()?.as_ref().unwrap().forward_flow(image)
    }
}
