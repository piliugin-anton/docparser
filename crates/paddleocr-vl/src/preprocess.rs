//! Image and prompt preprocessing aligned with HuggingFace PaddleOCR-VL.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use tokenizers::Tokenizer;

use crate::paddleocr_vl::Config;
use crate::VlmTask;

const PATCH_SIZE: usize = 14;
const SPATIAL_MERGE: usize = 2;
const FACTOR: usize = PATCH_SIZE * SPATIAL_MERGE;
/// From `preprocessor_config.json` on HF PaddleOCR-VL-1.6.
const MIN_PIXELS: usize = 112_896;
const MAX_PIXELS: usize = 1_003_520;

/// Resize so H,W are divisible by 28 and total pixels in [MIN, MAX] (`smart_resize` in HF image processor).
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    let mut h = height;
    let mut w = width;
    if h < FACTOR {
        w = ((w as f64 * FACTOR as f64) / h as f64).round() as usize;
        h = FACTOR;
    }
    if w < FACTOR {
        h = ((h as f64 * FACTOR as f64) / w as f64).round() as usize;
        w = FACTOR;
    }
    let aspect = if h > w {
        h as f64 / w as f64
    } else {
        w as f64 / h as f64
    };
    anyhow::ensure!(aspect <= 200.0, "aspect ratio {aspect} exceeds 200");

    let mut h_bar = ((h as f64 / FACTOR as f64).round() as usize) * FACTOR;
    let mut w_bar = ((w as f64 / FACTOR as f64).round() as usize) * FACTOR;
    let total = h_bar * w_bar;
    if total > MAX_PIXELS {
        let beta = ((h * w) as f64 / MAX_PIXELS as f64).sqrt();
        h_bar = ((h as f64 / beta / FACTOR as f64).floor() as usize) * FACTOR;
        w_bar = ((w as f64 / beta / FACTOR as f64).floor() as usize) * FACTOR;
    } else if total < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w) as f64).sqrt();
        h_bar = ((h as f64 * beta / FACTOR as f64).ceil() as usize) * FACTOR;
        w_bar = ((w as f64 * beta / FACTOR as f64).ceil() as usize) * FACTOR;
    }
    Ok((h_bar.max(FACTOR), w_bar.max(FACTOR)))
}

pub fn image_to_pixel_values(
    image: &RgbImage,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let (width, height) = image.dimensions();
    let (new_h, new_w) = smart_resize(height as usize, width as usize)?;
    let resized = image::imageops::resize(
        image,
        new_w as u32,
        new_h as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let mut normalized = vec![0f32; 3 * new_h * new_w];
    for c in 0..3 {
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x as u32, y as u32);
                let idx = c * new_h * new_w + y * new_w + x;
                normalized[idx] = p[c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }

    let pixel_values =
        Tensor::from_vec(normalized, (1, 3, new_h, new_w), device)?.to_dtype(dtype)?;
    let h_patches = (new_h / PATCH_SIZE) as u32;
    let w_patches = (new_w / PATCH_SIZE) as u32;
    let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;
    Ok((pixel_values, grid_thw))
}

pub fn num_image_tokens(grid_thw: &Tensor, spatial_merge: usize) -> Result<usize> {
    let g = grid_thw.to_vec2::<u32>()?;
    let h = g[0][1] as usize / spatial_merge;
    let w = g[0][2] as usize / spatial_merge;
    Ok(h * w)
}

const IMAGE_PLACEHOLDER: &str = "<|IMAGE_PLACEHOLDER|>";

/// Build `input_ids` the same way as HF `PaddleOCRVLProcessor` + `chat_template.jinja`.
pub fn build_input_ids(
    tokenizer: &Tokenizer,
    _cfg: &Config,
    task: VlmTask,
    num_image_tokens: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut text = format!(
        "<|begin_of_sentence|>User: <|IMAGE_START|>{IMAGE_PLACEHOLDER}<|IMAGE_END|>{task}\nAssistant:\n",
        task = task.prompt()
    );
    let expanded = IMAGE_PLACEHOLDER.repeat(num_image_tokens);
    text = text.replacen(IMAGE_PLACEHOLDER, &expanded, 1);
    let enc = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    Ok(Tensor::new(enc.get_ids(), device)?.unsqueeze(0)?)
}

pub fn load_tokenizer(model_dir: &std::path::Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))
}

pub fn eos_token_id(tokenizer: &Tokenizer) -> u32 {
    tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2)
}
