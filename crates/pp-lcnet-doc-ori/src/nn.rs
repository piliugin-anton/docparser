use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

pub fn hardswish(xs: &Tensor) -> Result<Tensor> {
    let six = Tensor::new(6f32, xs.device())?.broadcast_as(xs.shape())?;
    let three = Tensor::new(3f32, xs.device())?.broadcast_as(xs.shape())?;
    let relu6 = xs.add(&three)?.relu()?.minimum(&six)?;
    xs.mul(&relu6)?.div(&six)
}

pub fn hardsigmoid(xs: &Tensor) -> Result<Tensor> {
    let six = Tensor::new(6f32, xs.device())?.broadcast_as(xs.shape())?;
    let three = Tensor::new(3f32, xs.device())?.broadcast_as(xs.shape())?;
    let zero = Tensor::zeros(xs.shape(), xs.dtype(), xs.device())?;
    let one = Tensor::ones(xs.shape(), xs.dtype(), xs.device())?;
    xs.add(&three)?
        .relu()?
        .minimum(&six)?
        .div(&six)?
        .maximum(&zero)?
        .minimum(&one)
}

pub fn apply_activation(name: &str, xs: &Tensor) -> Result<Tensor> {
    match name {
        "hardswish" => hardswish(xs),
        "relu" => xs.relu(),
        _ => Ok(xs.clone()),
    }
}

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
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = kernel / 2;
        let cfg = Conv2dConfig {
            stride,
            padding: pad,
            groups,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("convolution"))?;
        let norm = BatchNorm2d::load(out_ch, vb.pp("normalization"))?;
        Ok(Self {
            conv,
            norm,
            act: act.to_string(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        apply_activation(&self.act, &x)
    }
}

pub struct SeModule {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl SeModule {
    pub fn load(ch: usize, reduction: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = (ch / reduction).max(1);
        let conv1 = candle_nn::conv2d(ch, hidden, 1, Default::default(), vb.pp("0"))?;
        let conv2 = candle_nn::conv2d(hidden, ch, 1, Default::default(), vb.pp("2"))?;
        Ok(Self { conv1, conv2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, _h, _w) = xs.dims4()?;
        let pooled = xs.mean_keepdim(2)?.mean_keepdim(3)?;
        let mut hidden = self.conv1.forward(&pooled)?.relu()?;
        hidden = self.conv2.forward(&hidden)?;
        hidden = hardsigmoid(&hidden)?;
        hidden.broadcast_mul(xs)
    }
}

pub struct DepthwiseSeparable {
    depthwise: ConvBnAct,
    se: Option<SeModule>,
    pointwise: ConvBnAct,
}

impl DepthwiseSeparable {
    pub fn load(
        spec: crate::config::BlockSpec,
        scale: f64,
        divisor: usize,
        hidden_act: &str,
        reduction: usize,
        cfg: &crate::config::PpLcnetConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_ch = crate::config::make_divisible(spec.in_channels as f64 * scale, divisor);
        let out_ch = crate::config::make_divisible(spec.out_channels as f64 * scale, divisor);
        let depthwise = ConvBnAct::new(
            in_ch,
            in_ch,
            spec.kernel,
            spec.stride,
            in_ch,
            hidden_act,
            vb.pp("depthwise_convolution"),
        )?;
        let se = if spec.use_se {
            Some(SeModule::load(
                in_ch,
                reduction,
                vb.pp("squeeze_excitation_module.convolutions"),
            )?)
        } else {
            None
        };
        let pointwise = ConvBnAct::new(
            in_ch,
            out_ch,
            1,
            1,
            1,
            hidden_act,
            vb.pp("pointwise_convolution"),
        )?;
        let _ = cfg;
        Ok(Self {
            depthwise,
            se,
            pointwise,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.depthwise.forward(xs)?;
        if let Some(se) = &self.se {
            x = se.forward(&x)?;
        }
        self.pointwise.forward(&x)
    }
}

pub struct PpLcnetModel {
    stem: ConvBnAct,
    blocks: Vec<Vec<DepthwiseSeparable>>,
    last_conv: Conv2d,
    head: Linear,
    hidden_act: String,
    hidden_dropout_prob: f32,
}

impl PpLcnetModel {
    pub fn load(cfg: &crate::config::PpLcnetConfig, vb: VarBuilder) -> Result<Self> {
        let stem_ch =
            crate::config::make_divisible(cfg.stem_channels as f64 * cfg.scale, cfg.divisor);
        let stem = ConvBnAct::new(
            3,
            stem_ch,
            3,
            cfg.stem_stride,
            1,
            &cfg.hidden_act,
            vb.pp("encoder.convolution"),
        )?;
        let mut blocks = Vec::new();
        for (stage_idx, stage) in cfg.block_configs.iter().enumerate() {
            let mut layers = Vec::new();
            for (layer_idx, spec) in stage.iter().enumerate() {
                layers.push(DepthwiseSeparable::load(
                    *spec,
                    cfg.scale,
                    cfg.divisor,
                    &cfg.hidden_act,
                    cfg.reduction,
                    cfg,
                    vb.pp(format!("encoder.blocks.{stage_idx}.layers.{layer_idx}")),
                )?);
            }
            blocks.push(layers);
        }
        let last_in = crate::config::make_divisible(
            cfg.block_configs
                .last()
                .and_then(|s| s.last())
                .map(|b| b.out_channels as f64)
                .unwrap_or(512.0)
                * cfg.scale,
            cfg.divisor,
        );
        let last_conv = candle_nn::conv2d_no_bias(
            last_in,
            cfg.class_expand,
            1,
            Default::default(),
            vb.pp("last_convolution"),
        )?;
        let head = candle_nn::linear(cfg.class_expand, cfg.num_labels(), vb.pp("head"))?;
        Ok(Self {
            stem,
            blocks,
            last_conv,
            head,
            hidden_act: cfg.hidden_act.clone(),
            hidden_dropout_prob: cfg.hidden_dropout_prob,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(pixel_values)?;
        candle_nn::ops::softmax_last_dim(&logits)
    }

    pub fn forward_logits(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.stem.forward(pixel_values)?;
        for stage in &self.blocks {
            for layer in stage {
                x = layer.forward(&x)?;
            }
        }
        let (_b, _c, _h, _w) = x.dims4()?;
        x = x.mean_keepdim(2)?.mean_keepdim(3)?;
        x = self.last_conv.forward(&x)?;
        x = apply_activation(&self.hidden_act, &x)?;
        let keep =
            Tensor::new(1.0 - self.hidden_dropout_prob, x.device())?.broadcast_as(x.shape())?;
        x = (x * keep)?;
        x = x.squeeze(2)?.squeeze(2)?;
        self.head.forward(&x)
    }
}
