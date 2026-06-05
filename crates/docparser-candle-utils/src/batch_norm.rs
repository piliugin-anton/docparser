//! Shared 2D batch normalization (inference mode, running stats).

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
}

impl BatchNorm2d {
    pub fn load(ch: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(ch, "weight")?,
            bias: vb.get(ch, "bias")?,
            running_mean: vb.get(ch, "running_mean")?,
            running_var: vb.get(ch, "running_var")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let eps = 1e-5f64;
        let w = self.weight.reshape((1, (), 1, 1))?;
        let b = self.bias.reshape((1, (), 1, 1))?;
        let rm = self.running_mean.reshape((1, (), 1, 1))?;
        let rv = self.running_var.reshape((1, (), 1, 1))?;
        let scale = (&w * (rv + eps)?.powf(-0.5)?)?;
        let bias = (&b - (&rm * &scale)?)?;
        x.broadcast_mul(&scale)?.broadcast_add(&bias)
    }
}
