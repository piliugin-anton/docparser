//! Bilinear `grid_sample` for deformable attention (align_corners=false).

use candle_core::{D, Result, Tensor};

/// `value`: (B*H, C, H, W), `grid`: (B*H, Q, P, 2) in [-1, 1].
/// Returns (B*H, C, Q, P).
pub fn grid_sample_bilinear(value: &Tensor, grid: &Tensor) -> Result<Tensor> {
    let (bh, c, h, w) = value.dims4()?;
    let (_bh2, q, p, two) = grid.dims4()?;
    if two != 2 {
        candle_core::bail!("grid last dim must be 2");
    }
    let v = value.to_dtype(candle_core::DType::F32)?;
    let g = grid.to_dtype(candle_core::DType::F32)?;
    let v_data = v.flatten_all()?.to_vec1::<f32>()?;
    let g_data = g.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; bh * c * q * p];
    for bhi in 0..bh {
        let v_base = bhi * c * h * w;
        let g_base = bhi * q * p * 2;
        let o_base = bhi * c * q * p;
        for qi in 0..q {
            for pi in 0..p {
                let gi = g_base + (qi * p + pi) * 2;
                let gx = g_data[gi];
                let gy = g_data[gi + 1];
                let ix = ((gx + 1.0) * 0.5 * (w as f32) - 0.5).clamp(0.0, (w - 1) as f32);
                let iy = ((gy + 1.0) * 0.5 * (h as f32) - 0.5).clamp(0.0, (h - 1) as f32);
                let x0 = ix.floor() as usize;
                let y0 = iy.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fx = ix - x0 as f32;
                let fy = iy - y0 as f32;
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w01 = fx * (1.0 - fy);
                let w10 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                for ci in 0..c {
                    let vb = v_base + ci * h * w;
                    let v00 = v_data[vb + y0 * w + x0];
                    let v01 = v_data[vb + y0 * w + x1];
                    let v10 = v_data[vb + y1 * w + x0];
                    let v11 = v_data[vb + y1 * w + x1];
                    out[o_base + ci * q * p + qi * p + pi] =
                        w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                }
            }
        }
    }
    Tensor::from_vec(out, (bh, c, q, p), value.device())?.to_dtype(value.dtype())
}

/// Deformable-DETR multi-scale attention (inference).
pub fn multiscale_deformable_attention(
    value: &Tensor,
    spatial_shapes: &[(usize, usize)],
    sampling_locations: &Tensor,
    attention_weights: &Tensor,
) -> Result<Tensor> {
    let (batch, _seq_len, n_heads, head_dim) = value.dims4()?;
    let dims = sampling_locations.dims();
    let (_b, num_queries, _nh2, n_levels, n_points, _two) =
        (dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
    let sampling_grids = ((sampling_locations * 2.0)? - 1.0)?;
    let mut sampling_value_list = Vec::with_capacity(n_levels);
    let mut offset = 0usize;
    for (level_id, &(height, width)) in spatial_shapes.iter().enumerate() {
        let n = height * width;
        let value_l = value.narrow(1, offset, n)?;
        offset += n;
        let value_l = value_l.flatten(2, 3)?.transpose(1, 2)?.reshape((
            batch * n_heads,
            head_dim,
            height,
            width,
        ))?;
        let grid_l = sampling_grids
            .narrow(3, level_id, 1)?
            .squeeze(3)?
            .transpose(1, 2)?
            .flatten(0, 1)?;
        sampling_value_list.push(grid_sample_bilinear(&value_l, &grid_l)?);
    }
    let stacked = Tensor::stack(&sampling_value_list, D::Minus1)?;
    let flat = stacked.flatten(3, 4)?;
    let attn = attention_weights.transpose(1, 2)?.reshape((
        batch * n_heads,
        1,
        num_queries,
        n_levels * n_points,
    ))?;
    let out = flat.broadcast_mul(&attn)?.sum(D::Minus1)?;
    out.reshape((batch, n_heads * head_dim, num_queries))?
        .transpose(1, 2)
}
