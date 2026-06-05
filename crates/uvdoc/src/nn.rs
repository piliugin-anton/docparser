use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::config::UvdocConfig;
use crate::padding::reflect_pad2d;

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

pub struct Prelu {
    weight: Tensor,
}

impl Prelu {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(1, "weight")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.reshape((1, (), 1, 1))?;
        let pos = x.relu()?;
        let neg = (x - pos.clone())?;
        pos + &neg.broadcast_mul(&w)?
    }
}

pub fn relu(xs: &Tensor) -> Result<Tensor> {
    xs.relu()
}

pub struct UvConv {
    conv: Conv2d,
    norm: BatchNorm2d,
    act: Option<Prelu>,
    use_relu: bool,
    reflect_pad: usize,
}

impl UvConv {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        activation: Option<&str>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_options(
            in_ch, out_ch, kernel, stride, padding, dilation, activation, false, 0, vb,
        )
    }

    pub fn new_with_options(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        activation: Option<&str>,
        bias: bool,
        reflect_pad: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride,
            padding: if reflect_pad > 0 { 0 } else { padding },
            dilation,
            ..Default::default()
        };
        let conv_vb = vb.pp("convolution");
        let conv = if bias {
            candle_nn::conv2d(in_ch, out_ch, kernel, cfg, conv_vb)?
        } else {
            candle_nn::conv2d_no_bias(in_ch, out_ch, kernel, cfg, conv_vb)?
        };
        let norm = BatchNorm2d::load(out_ch, vb.pp("normalization"))?;
        let (act, use_relu) = match activation {
            Some("prelu") => (Some(Prelu::load(vb.pp("activation"))?), false),
            Some("relu") | Some(_) => (None, true),
            None => (None, false),
        };
        Ok(Self {
            conv,
            norm,
            act,
            use_relu,
            reflect_pad,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.reflect_pad > 0 {
            reflect_pad2d(x, self.reflect_pad, self.reflect_pad)?
        } else {
            x.clone()
        };
        let mut x = self.conv.forward(&x)?;
        x = self.norm.forward(&x)?;
        if let Some(act) = &self.act {
            act.forward(&x)
        } else if self.use_relu {
            relu(&x)
        } else {
            Ok(x)
        }
    }
}

pub struct ReflectConv2d {
    conv: Conv2d,
    pad: usize,
}

impl ReflectConv2d {
    pub fn load(in_ch: usize, out_ch: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let pad = kernel / 2;
        let conv = candle_nn::conv2d(
            in_ch,
            out_ch,
            kernel,
            Conv2dConfig {
                padding: 0,
                ..Default::default()
            },
            vb,
        )?;
        Ok(Self { conv, pad })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = reflect_pad2d(x, self.pad, self.pad)?;
        self.conv.forward(&x)
    }
}

pub struct ResidualBlock {
    conv_down: Option<UvConv>,
    conv_start: UvConv,
    conv_final: UvConv,
}

impl ResidualBlock {
    pub fn load(
        in_ch: usize,
        out_ch: usize,
        dilation: usize,
        downsample: bool,
        kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_down = if downsample {
            Some(UvConv::new_with_options(
                in_ch,
                out_ch,
                kernel,
                2,
                kernel / 2,
                1,
                None,
                true,
                0,
                vb.pp("conv_down"),
            )?)
        } else {
            None
        };
        let conv_start = UvConv::new_with_options(
            in_ch,
            out_ch,
            kernel,
            if downsample { 2 } else { 1 },
            dilation * 2,
            dilation,
            Some("relu"),
            true,
            0,
            vb.pp("conv_start"),
        )?;
        let conv_final = UvConv::new_with_options(
            out_ch,
            out_ch,
            kernel,
            1,
            dilation * 2,
            dilation,
            None,
            true,
            0,
            vb.pp("conv_final"),
        )?;
        Ok(Self {
            conv_down,
            conv_start,
            conv_final,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(d) = &self.conv_down {
            d.forward(x)?
        } else {
            x.clone()
        };
        let mut x = self.conv_start.forward(x)?;
        x = self.conv_final.forward(&x)?;
        relu(&(x + residual)?)
    }
}

pub struct UvdocNet {
    resnet_head: Vec<UvConv>,
    resnet_stages: Vec<Vec<ResidualBlock>>,
    bridge: Vec<Vec<UvConv>>,
    bridge_connector: UvConv,
    conv_down: UvConv,
    conv_up: ReflectConv2d,
}

impl UvdocNet {
    pub fn load(cfg: &UvdocConfig, vb: VarBuilder) -> Result<Self> {
        let k = cfg.kernel_size;
        let bb = &cfg.backbone;
        let mut resnet_head = Vec::new();
        for (i, pair) in bb.resnet_head.iter().enumerate() {
            resnet_head.push(UvConv::new(
                pair[0],
                pair[1],
                k,
                2,
                k / 2,
                1,
                Some("relu"),
                vb.pp(format!("backbone.resnet.resnet_head.{i}")),
            )?);
        }
        let mut resnet_stages = Vec::new();
        for (si, stage) in bb.resnet_configs.iter().enumerate() {
            let mut layers = Vec::new();
            for (li, spec) in stage.iter().enumerate() {
                layers.push(ResidualBlock::load(
                    spec[0],
                    spec[1],
                    spec[2],
                    spec[3] == 1,
                    k,
                    vb.pp(format!("backbone.resnet.resnet_down.{si}.layers.{li}")),
                )?);
            }
            resnet_stages.push(layers);
        }
        let mut bridge = Vec::new();
        for (bi, stage) in bb.stage_configs.iter().enumerate() {
            let mut blocks = Vec::new();
            for (li, spec) in stage.iter().enumerate() {
                let in_ch = spec[0];
                let dilation = spec[1];
                blocks.push(UvConv::new(
                    in_ch,
                    in_ch,
                    3,
                    1,
                    dilation,
                    dilation,
                    Some("relu"),
                    vb.pp(format!("backbone.bridge.bridge.{bi}.blocks.{li}")),
                )?);
            }
            bridge.push(blocks);
        }
        let num_bridge = bb.stage_configs.len();
        let bridge_connector = UvConv::new(
            cfg.bridge_connector[0] * num_bridge,
            cfg.bridge_connector[1],
            1,
            1,
            0,
            1,
            Some("relu"),
            vb.pp("head.bridge_connector"),
        )?;
        let k = cfg.kernel_size;
        let head_pad = k / 2;
        let conv_down = UvConv::new_with_options(
            cfg.out_point_positions2d[0][0],
            cfg.out_point_positions2d[0][1],
            k,
            1,
            head_pad,
            1,
            Some("prelu"),
            false,
            head_pad,
            vb.pp("head.out_point_positions2D.conv_down"),
        )?;
        let conv_up = ReflectConv2d::load(
            cfg.out_point_positions2d[1][0],
            cfg.out_point_positions2d[1][1],
            k,
            vb.pp("head.out_point_positions2D.conv_up"),
        )?;
        Ok(Self {
            resnet_head,
            resnet_stages,
            bridge,
            bridge_connector,
            conv_down,
            conv_up,
        })
    }

    /// Flow field (B, 2, H, W) at head resolution — matches HF `last_hidden_state`.
    pub fn forward_flow(&self, image: &Tensor) -> Result<Tensor> {
        let mut x = image.clone();
        for head in &self.resnet_head {
            x = head.forward(&x)?;
        }
        for stage in &self.resnet_stages {
            for layer in stage {
                x = layer.forward(&x)?;
            }
        }
        let mut feature_maps = Vec::new();
        for blocks in &self.bridge {
            let mut y = x.clone();
            for block in blocks {
                y = block.forward(&y)?;
            }
            feature_maps.push(y);
        }
        x = Tensor::cat(&feature_maps, 1)?;
        x = self.bridge_connector.forward(&x)?;
        x = self.conv_down.forward(&x)?;
        x = self.conv_up.forward(&x)?;
        Ok(x)
    }

    /// Rectify using flow upsampled to the original image size (HF `post_process_document_rectification`).
    pub fn rectify_with_flow(&self, original_bgr: &Tensor, flow: &Tensor) -> Result<Tensor> {
        let (_b, _c, oh, ow) = original_bgr.dims4()?;
        let flow_up = crate::grid_sample::upsample_bilinear_align_corners(&flow, oh, ow)?;
        let grid = flow_up.permute((0, 2, 3, 1))?;
        crate::grid_sample::grid_sample_bilinear_align_corners(original_bgr, &grid)
    }
}
