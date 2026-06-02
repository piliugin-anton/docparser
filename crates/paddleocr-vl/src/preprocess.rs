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
const MIN_PIXELS: usize = 147_384;
const MAX_PIXELS: usize = 2_822_400;

/// Resize so H,W are divisible by 28 and total pixels in [MIN, MAX] (smart_resize).
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    let mut h = height;
    let mut w = width;
    if h < FACTOR {
        w = w * FACTOR / h.max(1);
        h = FACTOR;
    }
    if w < FACTOR {
        h = h * FACTOR / w.max(1);
        w = FACTOR;
    }
    let aspect = if h > w {
        h as f64 / w as f64
    } else {
        w as f64 / h as f64
    };
    anyhow::ensure!(aspect <= 200.0, "aspect ratio {aspect} exceeds 200");

    let mut h_bar = ((h + FACTOR / 2) / FACTOR) * FACTOR;
    let mut w_bar = ((w + FACTOR / 2) / FACTOR) * FACTOR;
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

pub fn build_input_ids(
    tokenizer: &Tokenizer,
    cfg: &Config,
    task: VlmTask,
    num_image_tokens: usize,
    device: &Device,
) -> Result<Tensor> {
    let bos = tokenizer
        .token_to_id("<|begin_of_sentence|>")
        .unwrap_or(1);
    let user = tokenizer
        .encode("User: ", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let task_enc = tokenizer
        .encode(task.prompt(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let assistant = tokenizer
        .encode("\nAssistant: ", false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut ids = vec![bos];
    ids.extend(user.get_ids());
    ids.push(cfg.vision_start_token_id);
    ids.extend(std::iter::repeat(cfg.image_token_id).take(num_image_tokens));
    ids.push(cfg.vision_end_token_id);
    ids.extend(task_enc.get_ids());
    ids.extend(assistant.get_ids());

    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
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
