//! Reading-order global pointer head.

use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;

pub struct GlobalPointer {
    head_size: usize,
    dense: Linear,
}

impl GlobalPointer {
    pub fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let head_size = 64;
        Ok(Self {
            head_size,
            dense: candle_nn::linear(cfg.d_model, head_size * 2, vb.pp("dense"))?,
        })
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, _) = inputs.dims3()?;
        let qk = self
            .dense
            .forward(inputs)?
            .reshape((batch_size, sequence_length, 2, self.head_size))?;
        let queries = qk.narrow(D::Minus2, 0, 1)?.squeeze(D::Minus2)?;
        let keys = qk.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
        let scale = (self.head_size as f64).sqrt();
        let logits = (&queries.matmul(&keys.transpose(1, 2)?)? / scale)?;
        let device = logits.device();
        let mut mask_data = vec![0f32; (sequence_length * sequence_length) as usize];
        for i in 0..sequence_length {
            for j in 0..sequence_length {
                if j <= i {
                    mask_data[i * sequence_length + j] = -1e4;
                }
            }
        }
        let mask = Tensor::from_vec(
            mask_data,
            (1, sequence_length, sequence_length),
            device,
        )?
        .to_dtype(logits.dtype())?;
        logits.broadcast_add(&mask)
    }
}
