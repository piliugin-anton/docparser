use candle_core::{Result, Tensor};

/// Bilinear resize with `align_corners=true` (matches Paddle UVDoc).
pub fn upsample_bilinear_align_corners(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (b, c, in_h, in_w) = x.dims4()?;
    if in_h == out_h && in_w == out_w {
        return Ok(x.clone());
    }
    let x = x.to_dtype(candle_core::DType::F32)?;
    let data = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; b * c * out_h * out_w];
    let scale_h = if out_h > 1 && in_h > 1 {
        (in_h - 1) as f64 / (out_h - 1) as f64
    } else {
        0.0
    };
    let scale_w = if out_w > 1 && in_w > 1 {
        (in_w - 1) as f64 / (out_w - 1) as f64
    } else {
        0.0
    };
    for bi in 0..b {
        for ci in 0..c {
            let base_in = (bi * c + ci) * in_h * in_w;
            let base_out = (bi * c + ci) * out_h * out_w;
            for ty in 0..out_h {
                for tx in 0..out_w {
                    let sy = (ty as f64 * scale_h).clamp(0.0, (in_h - 1) as f64);
                    let sx = (tx as f64 * scale_w).clamp(0.0, (in_w - 1) as f64);
                    let sy0 = sy.floor() as usize;
                    let sx0 = sx.floor() as usize;
                    let sy1 = (sy0 + 1).min(in_h - 1);
                    let sx1 = (sx0 + 1).min(in_w - 1);
                    let fy = (sy - sy0 as f64) as f32;
                    let fx = (sx - sx0 as f64) as f32;
                    let w00 = (1.0 - fy) * (1.0 - fx);
                    let w01 = (1.0 - fy) * fx;
                    let w10 = fy * (1.0 - fx);
                    let w11 = fy * fx;
                    let v00 = data[base_in + sy0 * in_w + sx0];
                    let v01 = data[base_in + sy0 * in_w + sx1];
                    let v10 = data[base_in + sy1 * in_w + sx0];
                    let v11 = data[base_in + sy1 * in_w + sx1];
                    out[base_out + ty * out_w + tx] =
                        w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_h, out_w), x.device())?.to_dtype(x.dtype())
}

/// `grid`: (B, H, W, 2) with values in [-1, 1], `align_corners=true`.
pub fn grid_sample_bilinear_align_corners(image: &Tensor, grid: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = image.dims4()?;
    let (_b2, gh, gw, two) = grid.dims4()?;
    if two != 2 {
        candle_core::bail!("grid last dim must be 2");
    }
    let img = image.to_dtype(candle_core::DType::F32)?;
    let grid = grid.to_dtype(candle_core::DType::F32)?;
    let img_data = img.flatten_all()?.to_vec1::<f32>()?;
    let grid_data = grid.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; b * c * gh * gw];
    for bi in 0..b {
        let img_base = bi * c * h * w;
        let grid_base = bi * gh * gw * 2;
        let out_base = bi * c * gh * gw;
        for gy in 0..gh {
            for gx in 0..gw {
                let gi = grid_base + (gy * gw + gx) * 2;
                let gx_val = grid_data[gi];
                let gy_val = grid_data[gi + 1];
                let sx = if w > 1 {
                    ((gx_val + 1.0) * 0.5 * (w - 1) as f32).clamp(0.0, (w - 1) as f32)
                } else {
                    0.0
                };
                let sy = if h > 1 {
                    ((gy_val + 1.0) * 0.5 * (h - 1) as f32).clamp(0.0, (h - 1) as f32)
                } else {
                    0.0
                };
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                let w00 = (1.0 - fy) * (1.0 - fx);
                let w01 = (1.0 - fy) * fx;
                let w10 = fy * (1.0 - fx);
                let w11 = fy * fx;
                for ci in 0..c {
                    let ch_base = img_base + ci * h * w;
                    let v00 = img_data[ch_base + y0 * w + x0];
                    let v01 = img_data[ch_base + y0 * w + x1];
                    let v10 = img_data[ch_base + y1 * w + x0];
                    let v11 = img_data[ch_base + y1 * w + x1];
                    out[out_base + ci * gh * gw + gy * gw + gx] =
                        w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, gh, gw), image.device())?.to_dtype(image.dtype())
}
