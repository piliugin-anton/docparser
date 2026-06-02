//! HGNet-V2 backbone (`model.backbone.model.*`).

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use super::config::HgNetV2Config;
use super::nn::{activation, pad_chw, max_pool2d_ceil, ConvBnAct, FrozenConvBnAct};

pub struct HgNetV2Backbone {
    embedder: HgNetV2Embeddings,
    encoder: HgNetV2Encoder,
}

impl HgNetV2Backbone {
    pub fn new(cfg: &HgNetV2Config, frozen_bn: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embedder: HgNetV2Embeddings::new(cfg, frozen_bn, vb.pp("embedder"))?,
            encoder: HgNetV2Encoder::new(cfg, frozen_bn, vb.pp("encoder"))?,
        })
    }

    /// Returns feature maps for stages in `out_features` order: stage1..stage4.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.embedder.forward(pixel_values)?;
        self.encoder.forward_features(&x)
    }
}

struct HgNetV2Embeddings {
    stem1: ConvBlock,
    stem2a: ConvBlock,
    stem2b: ConvBlock,
    stem3: ConvBlock,
    stem4: ConvBlock,
    act: String,
}

enum ConvBlock {
    Train(ConvBnAct),
    Frozen(FrozenConvBnAct),
}

impl ConvBlock {
    fn new(
        cfg: &HgNetV2Config,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        act: Option<&str>,
        frozen: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let act_name = match act {
            Some(a) => a,
            None => cfg.hidden_act.as_str(),
        };
        if frozen {
            Ok(Self::Frozen(FrozenConvBnAct::new(
                in_ch, out_ch, kernel, stride, groups, act_name, vb,
            )?))
        } else {
            Ok(Self::Train(ConvBnAct::new(
                in_ch, out_ch, kernel, stride, groups, act_name, 1e-5, vb,
            )?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Train(m) => m.forward(x),
            Self::Frozen(m) => m.forward(x),
        }
    }
}

impl HgNetV2Embeddings {
    fn new(cfg: &HgNetV2Config, frozen: bool, vb: VarBuilder) -> Result<Self> {
        let s = &cfg.stem_channels;
        let st = &cfg.stem_strides;
        Ok(Self {
            stem1: ConvBlock::new(cfg, s[0], s[1], 3, st[0], 1, Some(&cfg.hidden_act), frozen, vb.pp("stem1"))?,
            stem2a: ConvBlock::new(cfg, s[1], s[1] / 2, 2, st[1], 1, Some(&cfg.hidden_act), frozen, vb.pp("stem2a"))?,
            stem2b: ConvBlock::new(cfg, s[1] / 2, s[1], 2, st[2], 1, Some(&cfg.hidden_act), frozen, vb.pp("stem2b"))?,
            stem3: ConvBlock::new(cfg, s[1] * 2, s[1], 3, st[3], 1, Some(&cfg.hidden_act), frozen, vb.pp("stem3"))?,
            stem4: ConvBlock::new(cfg, s[1], s[2], 1, st[4], 1, Some(&cfg.hidden_act), frozen, vb.pp("stem4"))?,
            act: cfg.hidden_act.clone(),
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut embedding = self.stem1.forward(pixel_values)?;
        embedding = pad_chw(&embedding, 1, 1)?;
        let emb_stem_2a = self.stem2a.forward(&embedding)?;
        let emb_stem_2a = pad_chw(&emb_stem_2a, 1, 1)?;
        let emb_stem_2a = self.stem2b.forward(&emb_stem_2a)?;
        let pooled = max_pool2d_ceil(&embedding, 2, 1)?;
        let embedding = Tensor::cat(&[&pooled, &emb_stem_2a], 1)?;
        let embedding = self.stem3.forward(&embedding)?;
        self.stem4.forward(&embedding)
    }
}

struct HgNetV2Encoder {
    stages: Vec<HgNetV2Stage>,
    out_indices: Vec<usize>,
}

impl HgNetV2Encoder {
    fn new(cfg: &HgNetV2Config, frozen: bool, vb: VarBuilder) -> Result<Self> {
        let n = cfg.stage_in_channels.len();
        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            stages.push(HgNetV2Stage::new(cfg, i, frozen, vb.pp(format!("stages.{i}")))?);
        }
        let out_indices: Vec<usize> = (0..n).map(|i| i + 1).collect();
        Ok(Self { stages, out_indices })
    }

    fn forward_features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut hidden = x.clone();
        let mut states = vec![hidden.clone()];
        for stage in &self.stages {
            hidden = stage.forward(&hidden)?;
            states.push(hidden.clone());
        }
        Ok(self.out_indices.iter().map(|&i| states[i].clone()).collect())
    }
}

struct HgNetV2Stage {
    downsample: Downsample,
    blocks: Vec<HgNetV2BasicLayer>,
}

enum Downsample {
    None,
    Conv(ConvBlock),
}

impl HgNetV2Stage {
    fn new(cfg: &HgNetV2Config, stage_index: usize, frozen: bool, vb: VarBuilder) -> Result<Self> {
        let in_ch = cfg.stage_in_channels[stage_index];
        let downsample = if cfg.stage_downsample[stage_index] {
            Downsample::Conv(ConvBlock::new(
                cfg,
                in_ch,
                in_ch,
                3,
                cfg.stage_downsample_strides[stage_index],
                in_ch,
                Some("identity"),
                frozen,
                vb.pp("downsample"),
            )?)
        } else {
            Downsample::None
        };
        let num_blocks = cfg.stage_num_blocks[stage_index];
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let in_c = if i == 0 { in_ch } else { cfg.stage_out_channels[stage_index] };
            blocks.push(HgNetV2BasicLayer::new(
                cfg,
                in_c,
                cfg.stage_mid_channels[stage_index],
                cfg.stage_out_channels[stage_index],
                cfg.stage_numb_of_layers[stage_index],
                cfg.stage_kernel_size[stage_index],
                i != 0,
                cfg.stage_light_block[stage_index],
                frozen,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self { downsample, blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = match &self.downsample {
            Downsample::None => x.clone(),
            Downsample::Conv(d) => d.forward(x)?,
        };
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

struct HgNetV2BasicLayer {
    layers: Vec<BasicConv>,
    aggregation: (ConvBlock, ConvBlock),
    residual: bool,
}

enum BasicConv {
    Standard(ConvBlock),
    Light(LightBlock),
}

struct LightBlock {
    conv1: ConvBlock,
    conv2: ConvBlock,
}

impl LightBlock {
    fn new(cfg: &HgNetV2Config, in_ch: usize, out_ch: usize, kernel: usize, frozen: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: ConvBlock::new(cfg, in_ch, out_ch, 1, 1, 1, Some("identity"), frozen, vb.pp("conv1"))?,
            conv2: ConvBlock::new(cfg, out_ch, out_ch, kernel, 1, out_ch, None, frozen, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        self.conv2.forward(&x)
    }
}

impl HgNetV2BasicLayer {
    fn new(
        cfg: &HgNetV2Config,
        in_channels: usize,
        middle_channels: usize,
        out_channels: usize,
        layer_num: usize,
        kernel_size: usize,
        residual: bool,
        light_block: bool,
        frozen: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_num);
        for i in 0..layer_num {
            let in_c = if i == 0 { in_channels } else { middle_channels };
            let block = if light_block {
                BasicConv::Light(LightBlock::new(cfg, in_c, middle_channels, kernel_size, frozen, vb.pp(format!("layers.{i}")))?)
            } else {
                BasicConv::Standard(ConvBlock::new(
                    cfg, in_c, middle_channels, kernel_size, 1, 1, Some(&cfg.hidden_act), frozen, vb.pp(format!("layers.{i}")),
                )?)
            };
            layers.push(block);
        }
        let total = in_channels + layer_num * middle_channels;
        Ok(Self {
            layers,
            aggregation: (
                ConvBlock::new(cfg, total, out_channels / 2, 1, 1, 1, Some(&cfg.hidden_act), frozen, vb.pp("aggregation.0"))?,
                ConvBlock::new(cfg, out_channels / 2, out_channels, 1, 1, 1, Some(&cfg.hidden_act), frozen, vb.pp("aggregation.1"))?,
            ),
            residual,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let mut outputs = vec![x.clone()];
        for layer in &self.layers {
            let h = match layer {
                BasicConv::Standard(c) => c.forward(outputs.last().unwrap())?,
                BasicConv::Light(c) => c.forward(outputs.last().unwrap())?,
            };
            outputs.push(h.clone());
        }
        let cat = Tensor::cat(&outputs, 1)?;
        let mut h = self.aggregation.0.forward(&cat)?;
        h = self.aggregation.1.forward(&h)?;
        if self.residual {
            h = (&h + &identity)?;
        }
        Ok(h)
    }
}
