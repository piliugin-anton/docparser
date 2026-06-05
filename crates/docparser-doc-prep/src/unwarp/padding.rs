use candle_core::{Result, Tensor};

/// PyTorch `F.pad(..., mode="reflect")` on the last two dimensions.
pub fn reflect_pad2d(x: &Tensor, pad_h: usize, pad_w: usize) -> Result<Tensor> {
    if pad_h == 0 && pad_w == 0 {
        return Ok(x.clone());
    }
    let (b, c, h, w) = x.dims4()?;
    let out_h = h + 2 * pad_h;
    let out_w = w + 2 * pad_w;
    let x = x.to_dtype(candle_core::DType::F32)?;
    let data = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; b * c * out_h * out_w];
    for bi in 0..b {
        for ci in 0..c {
            let base_in = (bi * c + ci) * h * w;
            let base_out = (bi * c + ci) * out_h * out_w;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let iy = reflect_index(oy as i64 - pad_h as i64, h);
                    let ix = reflect_index(ox as i64 - pad_w as i64, w);
                    out[base_out + oy * out_w + ox] = data[base_in + iy * w + ix];
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_h, out_w), x.device())?.to_dtype(x.dtype())
}

fn reflect_index(mut idx: i64, size: usize) -> usize {
    if size <= 1 {
        return 0;
    }
    let size = size as i64;
    while idx < 0 || idx >= size {
        if idx < 0 {
            idx = -idx;
        } else {
            idx = 2 * size - idx - 2;
        }
    }
    idx as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    fn reflect_pad_matches_pytorch_pattern() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let x = Tensor::from_vec(data, (1, 1, 3, 4), &device).unwrap();
        let p = reflect_pad2d(&x, 2, 2).unwrap();
        assert_eq!(p.dims4().unwrap(), (1, 1, 7, 8));
        let v = p.i((0, 0, 0, 0)).unwrap().to_scalar::<f32>().unwrap();
        assert!((v - 10.0).abs() < 1e-6);
    }
}
