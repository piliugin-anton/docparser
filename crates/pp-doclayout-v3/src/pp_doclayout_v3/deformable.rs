//! Multi-scale deformable attention (decoder cross-attention).

use candle_core::{D, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;
use super::grid_sample::multiscale_deformable_attention;

pub struct MultiscaleDeformableAttention {
    d_model: usize,
    n_levels: usize,
    n_heads: usize,
    n_points: usize,
    sampling_offsets: Linear,
    attention_weights: Linear,
    value_proj: Linear,
    output_proj: Linear,
}

impl MultiscaleDeformableAttention {
    pub fn new(
        cfg: &PpDocLayoutV3Config,
        num_heads: usize,
        n_points: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.d_model;
        let n_levels = cfg.num_feature_levels;
        Ok(Self {
            d_model,
            n_levels,
            n_heads: num_heads,
            n_points,
            sampling_offsets: candle_nn::linear(
                d_model,
                num_heads * n_levels * n_points * 2,
                vb.pp("sampling_offsets"),
            )?,
            attention_weights: candle_nn::linear(
                d_model,
                num_heads * n_levels * n_points,
                vb.pp("attention_weights"),
            )?,
            value_proj: candle_nn::linear(d_model, d_model, vb.pp("value_proj"))?,
            output_proj: candle_nn::linear(d_model, d_model, vb.pp("output_proj"))?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        position_embeddings: Option<&Tensor>,
        reference_points: &Tensor,
        spatial_shapes: &[(usize, usize)],
    ) -> Result<Tensor> {
        let mut hs = hidden_states.clone();
        if let Some(pe) = position_embeddings {
            hs = (&hs + pe)?;
        }
        let (batch_size, num_queries, _) = hs.dims3()?;
        let (_b, sequence_length, _) = encoder_hidden_states.dims3()?;
        let value = self.value_proj.forward(encoder_hidden_states)?;
        let value = value.reshape((
            batch_size,
            sequence_length,
            self.n_heads,
            self.d_model / self.n_heads,
        ))?;
        let sampling_offsets = self.sampling_offsets.forward(&hs)?.reshape((
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        ))?;
        let attention_weights = self.attention_weights.forward(&hs)?.reshape((
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_points,
        ))?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&attention_weights)?.reshape((
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
        ))?;
        let num_coords = reference_points.dims()[reference_points.dims().len() - 1];
        let sampling_locations = if num_coords == 2 {
            let mut norm_flat = Vec::with_capacity(spatial_shapes.len() * 2);
            for &(h, w) in spatial_shapes {
                norm_flat.push(w as f32);
                norm_flat.push(h as f32);
            }
            let norm_t = Tensor::from_vec(norm_flat, (1, 1, 1, self.n_levels, 1, 2), hs.device())?;
            let mut rp = reference_points.unsqueeze(2)?;
            rp = rp.unsqueeze(4)?;
            let off = sampling_offsets.broadcast_div(&norm_t)?;
            let rp = rp.broadcast_as(off.shape())?;
            (&rp + &off)?
        } else {
            let mut rp = reference_points.narrow(D::Minus1, 0, 2)?;
            let mut scale = reference_points.narrow(D::Minus1, 2, 2)?;
            // reference_points is [B, Q, 1, 4] after decoder unsqueeze(2)
            rp = rp.unsqueeze(3)?.unsqueeze(3)?;
            scale = scale.unsqueeze(3)?.unsqueeze(3)?;
            let n_pts = Tensor::new(self.n_points as f32, hs.device())?
                .to_dtype(sampling_offsets.dtype())?;
            let off = (sampling_offsets
                .broadcast_div(&n_pts)?
                .broadcast_mul(&scale)?
                * 0.5)?;
            let rp = rp.broadcast_as(off.shape())?;
            (&rp + &off)?
        };
        let output = multiscale_deformable_attention(
            &value,
            spatial_shapes,
            &sampling_locations,
            &attention_weights,
        )?;
        self.output_proj.forward(&output)
    }
}
