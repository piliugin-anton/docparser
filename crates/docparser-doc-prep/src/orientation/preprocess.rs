use super::{DocOriError, Result};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub crop_size: u32,
    pub resize_short: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PreprocessorConfig {
    pub fn from_dir(model_dir: &std::path::Path) -> Result<Self> {
        let data =
            docparser_candle_utils::read_json_from_dir(model_dir, "preprocessor_config.json")?;
        #[derive(Deserialize)]
        struct Root {
            crop_size: Option<u32>,
            resize_short: Option<u32>,
            image_mean: Option<Vec<f32>>,
            image_std: Option<Vec<f32>>,
        }
        let root: Root = serde_json::from_str(&data)?;
        let mean = root
            .image_mean
            .and_then(|v| v.try_into().ok())
            .unwrap_or([0.406, 0.456, 0.485]);
        let std = root
            .image_std
            .and_then(|v| v.try_into().ok())
            .unwrap_or([0.225, 0.224, 0.229]);
        Ok(Self {
            crop_size: root.crop_size.unwrap_or(224),
            resize_short: root.resize_short.unwrap_or(256),
            mean,
            std,
        })
    }
}

/// Matches HF `PPLCNetImageProcessor`: bilinear resize (PIL), center crop, normalize RGB, swap to BGR.
pub fn preprocess(image: &RgbImage, cfg: &PreprocessorConfig, device: &Device) -> Result<Tensor> {
    let (w, h) = image.dimensions();
    let short = h.min(w) as f32;
    let scale = cfg.resize_short as f32 / short;
    let new_w = (w as f32 * scale).round().max(1.0) as u32;
    let new_h = (h as f32 * scale).round().max(1.0) as u32;
    let resized = resize_bilinear(image, new_w, new_h)?;
    let cropped = center_crop(&resized, cfg.crop_size, cfg.crop_size);

    let crop = cfg.crop_size as usize;
    let mut data = vec![0f32; 3 * crop * crop];
    for y in 0..crop {
        for x in 0..crop {
            let p = cropped.get_pixel(x as u32, y as u32);
            // Normalize RGB channels with ImageNet stats, then store as BGR (HF order).
            let r = (p[0] as f32 / 255.0 - cfg.mean[0]) / cfg.std[0];
            let g = (p[1] as f32 / 255.0 - cfg.mean[1]) / cfg.std[1];
            let b = (p[2] as f32 / 255.0 - cfg.mean[2]) / cfg.std[2];
            let planes = [b, g, r];
            for c in 0..3 {
                let idx = c * crop * crop + y * crop + x;
                data[idx] = planes[c];
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, crop, crop), device)
        .map_err(DocOriError::Candle)?
        .to_dtype(DType::F32)
        .map_err(DocOriError::Candle)?)
}

fn resize_bilinear(image: &RgbImage, target_w: u32, target_h: u32) -> Result<RgbImage> {
    let (src_w, src_h) = image.dimensions();
    if src_w == target_w && src_h == target_h {
        return Ok(image.clone());
    }
    // `FilterType::Triangle` matches PIL/torchvision bilinear more closely than fast_image_resize here.
    Ok(image::imageops::resize(
        image,
        target_w,
        target_h,
        image::imageops::FilterType::Triangle,
    ))
}

fn center_crop(image: &RgbImage, crop_h: u32, crop_w: u32) -> RgbImage {
    let (mut w, mut h) = image.dimensions();
    let mut img = image.clone();
    if crop_w > w || crop_h > h {
        let pad_l = if crop_w > w { (crop_w - w) / 2 } else { 0 };
        let pad_t = if crop_h > h { (crop_h - h) / 2 } else { 0 };
        let pad_r = if crop_w > w { (crop_w - w + 1) / 2 } else { 0 };
        let pad_b = if crop_h > h { (crop_h - h + 1) / 2 } else { 0 };
        img = pad_rgb(&img, pad_l, pad_t, pad_r, pad_b);
        w = img.width();
        h = img.height();
    }
    let x0 = ((w - crop_w) / 2).max(0);
    let y0 = ((h - crop_h) / 2).max(0);
    image::imageops::crop_imm(&img, x0, y0, crop_w.min(w), crop_h.min(h)).to_image()
}

fn pad_rgb(image: &RgbImage, left: u32, top: u32, right: u32, bottom: u32) -> RgbImage {
    let (w, h) = image.dimensions();
    let out_w = w + left + right;
    let out_h = h + top + bottom;
    let mut out = RgbImage::new(out_w, out_h);
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(x + left, y + top, *image.get_pixel(x, y));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_short_keeps_aspect_ratio() {
        let img = RgbImage::new(506, 378);
        let cfg = PreprocessorConfig {
            crop_size: 224,
            resize_short: 256,
            mean: [0.406, 0.456, 0.485],
            std: [0.225, 0.224, 0.229],
        };
        let short = img.height().min(img.width()) as f32;
        let scale = cfg.resize_short as f32 / short;
        let expected_w = (506.0 * scale).round() as u32;
        let expected_h = (378.0 * scale).round() as u32;
        let resized = resize_bilinear(&img, expected_w, expected_h).unwrap();
        assert_eq!(resized.dimensions(), (expected_w, expected_h));
    }
}
