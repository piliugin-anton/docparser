//! Shared layers (conv+bn, frozen bn, MLP, activations).

use candle_core::{Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;

pub fn activation(name: &str, xs: &Tensor) -> Result<Tensor> {
    match name {
        "relu" => xs.relu(),
        "gelu" => xs.gelu(),
        "silu" | "swish" => xs.silu(),
        "identity" | "none" => Ok(xs.clone()),
        _ => Ok(xs.clone()),
    }
}

/// Frozen BatchNorm2d (backbone).
pub struct FrozenBatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
}

impl FrozenBatchNorm2d {
    pub fn load(n: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(n, "weight")?,
            bias: vb.get(n, "bias")?,
            running_mean: vb.get(n, "running_mean")?,
            running_var: vb.get(n, "running_var")?,
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

pub struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm2d {
    pub fn load(n: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(n, "weight")?,
            bias: vb.get(n, "bias")?,
            running_mean: vb.get(n, "running_mean")?,
            running_var: vb.get(n, "running_var")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.reshape((1, (), 1, 1))?;
        let b = self.bias.reshape((1, (), 1, 1))?;
        let rm = self.running_mean.reshape((1, (), 1, 1))?;
        let rv = self.running_var.reshape((1, (), 1, 1))?;
        let scale = (&w * (rv + self.eps)?.powf(-0.5)?)?;
        let bias = (&b - (&rm * &scale)?)?;
        x.broadcast_mul(&scale)?.broadcast_add(&bias)
    }
}

pub struct ConvBnAct {
    conv: Conv2d,
    norm: BatchNorm2d,
    act: String,
}

impl ConvBnAct {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        act: &str,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = (kernel - 1) / 2;
        let cfg = Conv2dConfig {
            stride,
            padding: pad,
            groups,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("convolution"))?;
        let norm = BatchNorm2d::load(out_ch, eps, vb.pp("normalization"))?;
        Ok(Self {
            conv,
            norm,
            act: act.to_string(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        activation(&self.act, &x)
    }
}

pub struct FrozenConvBnAct {
    conv: Conv2d,
    norm: FrozenBatchNorm2d,
    act: String,
}

impl FrozenConvBnAct {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        act: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = (kernel - 1) / 2;
        let cfg = Conv2dConfig {
            stride,
            padding: pad,
            groups,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("convolution"))?;
        let norm = FrozenBatchNorm2d::load(out_ch, vb.pp("normalization"))?;
        Ok(Self {
            conv,
            norm,
            act: act.to_string(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        activation(&self.act, &x)
    }
}

pub struct ConvNormLayer {
    conv: Conv2d,
    norm: BatchNorm2d,
    act: Option<String>,
}

impl ConvNormLayer {
    pub fn new(
        cfg: &PpDocLayoutV3Config,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: Option<usize>,
        act: Option<&str>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = padding.unwrap_or(kernel / 2);
        let conv_cfg = Conv2dConfig {
            stride,
            padding: pad,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, conv_cfg, vb.pp("conv"))?;
        let norm = BatchNorm2d::load(out_ch, cfg.batch_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            conv,
            norm,
            act: act.map(str::to_string),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        match &self.act {
            Some(a) => activation(a, &x),
            None => Ok(x),
        }
    }
}

pub struct ConvLayer {
    conv: Conv2d,
    norm: BatchNorm2d,
    act: String,
}

impl ConvLayer {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        act: &str,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = (kernel - 1) / 2;
        let cfg = Conv2dConfig {
            stride,
            padding: pad,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("convolution"))?;
        let norm = BatchNorm2d::load(out_ch, eps, vb.pp("normalization"))?;
        Ok(Self {
            conv,
            norm,
            act: act.to_string(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        activation(&self.act, &x)
    }
}

pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    act: String,
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, act: &str, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(hidden, intermediate, vb.pp("fc1"))?,
            fc2: candle_nn::linear(intermediate, hidden, vb.pp("fc2"))?,
            act: act.to_string(),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = activation(&self.act, &xs)?;
        self.fc2.forward(&xs)
    }
}

pub struct MlpPredictionHead {
    layers: Vec<Linear>,
    num_layers: usize,
}

impl MlpPredictionHead {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let mut dims = vec![input_dim];
        for _ in 0..num_layers - 1 {
            dims.push(hidden_dim);
        }
        dims.push(output_dim);
        for i in 0..num_layers {
            layers.push(candle_nn::linear(dims[i], dims[i + 1], vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self { layers, num_layers })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.num_layers - 1 {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::LayerNormConfig { eps, ..Default::default() };
        Ok(Self {
            inner: candle_nn::layer_norm(size, cfg, vb)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

pub fn linear_b(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    candle_nn::linear_b(in_dim, out_dim, true, vb)
}

/// Nearest-neighbor 2× upsample (NCHW).
pub fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h, 1, w, 1))?
        .repeat((1, 1, 1, 2, 1, 2))?
        .reshape((b, c, h * 2, w * 2))
}

/// Bilinear upsample to target size, align_corners=false.
pub fn upsample_bilinear(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (b, c, in_h, in_w) = x.dims4()?;
    if in_h == out_h && in_w == out_w {
        return Ok(x.clone());
    }
    let x = x.to_dtype(candle_core::DType::F32)?;
    let scale_h = in_h as f64 / out_h as f64;
    let scale_w = in_w as f64 / out_w as f64;
    let mut out = vec![0f32; b * c * out_h * out_w];
    let data = x.flatten_all()?.to_vec1::<f32>()?;
    for bi in 0..b {
        for ci in 0..c {
            let base_in = (bi * c + ci) * in_h * in_w;
            let base_out = (bi * c + ci) * out_h * out_w;
            for ty in 0..out_h {
                for tx in 0..out_w {
                    let sy = ((ty as f64 + 0.5) * scale_h - 0.5).clamp(0.0, (in_h - 1) as f64);
                    let sx = ((tx as f64 + 0.5) * scale_w - 0.5).clamp(0.0, (in_w - 1) as f64);
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

pub fn pad_chw(x: &Tensor, pad_w: usize, pad_h: usize) -> Result<Tensor> {
    // pad right and bottom by (pad_w, pad_h)
    let (b, c, h, w) = x.dims4()?;
    let device = x.device();
    let zeros = Tensor::zeros((b, c, h, pad_w), x.dtype(), device)?;
    let x = Tensor::cat(&[x, &zeros], D::Minus1)?;
    let zeros = Tensor::zeros((b, c, pad_h, w + pad_w), x.dtype(), device)?;
    Tensor::cat(&[x, zeros], D::Minus2)
}

pub fn max_pool2d_ceil(x: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    let out_h = if h >= kernel { (h - kernel) / stride + 1 } else { 0 };
    let out_w = if w >= kernel { (w - kernel) / stride + 1 } else { 0 };
    let mut outs = Vec::new();
    let data = x.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (b, c, h, w) = x.dims4()?;
    for bi in 0..b {
        for ci in 0..c {
            let base = (bi * c + ci) * h * w;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut max_v = f32::NEG_INFINITY;
                    for ky in 0..kernel {
                        let iy = oy * stride + ky;
                        if iy >= h {
                            continue;
                        }
                        for kx in 0..kernel {
                            let ix = ox * stride + kx;
                            if ix >= w {
                                continue;
                            }
                            max_v = max_v.max(data[base + iy * w + ix]);
                        }
                    }
                    outs.push(max_v);
                }
            }
        }
    }
    Tensor::from_vec(outs, (b, c, out_h, out_w), x.device())?.to_dtype(x.dtype())
}
