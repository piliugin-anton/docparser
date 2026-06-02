//! Hybrid encoder + conv backbone wrapper.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;
use super::hgnet_v2::HgNetV2Backbone;
use super::nn::{
    activation, linear_b, upsample_bilinear, upsample_nearest_2x, BatchNorm2d, ConvLayer,
    ConvNormLayer, LayerNorm, Mlp,
};

pub struct ConvEncoder {
    backbone: HgNetV2Backbone,
}

impl ConvEncoder {
    pub fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let hg = cfg.hgnet();
        let backbone = HgNetV2Backbone::new(&hg, cfg.freeze_backbone_batch_norms, vb.pp("model"))?;
        Ok(Self { backbone })
    }

    pub fn forward(&self, pixel_values: &Tensor, _pixel_mask: &Tensor) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let maps = self.backbone.forward(pixel_values)?;
        let x4 = maps[0].clone();
        let mut out = Vec::new();
        for m in maps {
            out.push((m, Tensor::ones((1,), pixel_values.dtype(), pixel_values.device())?));
        }
        Ok((x4, out))
    }
}

pub struct EncoderInputProj {
    layers: Vec<(Conv2d, BatchNorm2d)>,
}

impl EncoderInputProj {
    pub fn new(cfg: &PpDocLayoutV3Config, in_channels: &[usize], vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, &in_ch) in in_channels.iter().enumerate() {
            let conv = candle_nn::conv2d_no_bias(
                in_ch,
                cfg.encoder_hidden_dim,
                1,
                Conv2dConfig::default(),
                vb.pp(format!("{i}.0")),
            )?;
            let norm = BatchNorm2d::load(cfg.encoder_hidden_dim, cfg.batch_norm_eps, vb.pp(format!("{i}.1")))?;
            layers.push((conv, norm));
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, i: usize) -> Result<Tensor> {
        let x = self.layers[i].0.forward(x)?;
        self.layers[i].1.forward(&x)
    }
}

pub struct HybridEncoderOutput {
    pub feature_maps: Vec<Tensor>,
    pub mask_feat: Tensor,
}

pub struct HybridEncoder {
    aifi: Vec<AifiLayer>,
    encode_proj_layers: Vec<usize>,
    lateral_convs: Vec<ConvNormLayer>,
    fpn_blocks: Vec<CspRepLayer>,
    downsample_convs: Vec<ConvNormLayer>,
    pan_blocks: Vec<CspRepLayer>,
    mask_feature_head: MaskFeatFpn,
    encoder_mask_lateral: ConvLayer,
    encoder_mask_output: EncoderMaskOutput,
}

impl HybridEncoder {
    pub fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let n_fpn = cfg.encoder_in_channels.len() - 1;
        let mut lateral_convs = Vec::new();
        let mut fpn_blocks = Vec::new();
        for i in 0..n_fpn {
            lateral_convs.push(ConvNormLayer::new(
                cfg,
                cfg.encoder_hidden_dim,
                cfg.encoder_hidden_dim,
                1,
                1,
                None,
                Some(&cfg.activation_function),
                vb.pp(format!("lateral_convs.{i}")),
            )?);
            fpn_blocks.push(CspRepLayer::new(cfg, vb.pp(format!("fpn_blocks.{i}")))?);
        }
        let mut downsample_convs = Vec::new();
        let mut pan_blocks = Vec::new();
        for i in 0..n_fpn {
            downsample_convs.push(ConvNormLayer::new(
                cfg,
                cfg.encoder_hidden_dim,
                cfg.encoder_hidden_dim,
                3,
                2,
                None,
                Some(&cfg.activation_function),
                vb.pp(format!("downsample_convs.{i}")),
            )?);
            pan_blocks.push(CspRepLayer::new(cfg, vb.pp(format!("pan_blocks.{i}")))?);
        }
        let n_aifi = cfg.encode_proj_layers.len();
        let mut aifi = Vec::new();
        for i in 0..n_aifi {
            aifi.push(AifiLayer::new(cfg, vb.pp(format!("encoder.{i}")))?);
        }
        Ok(Self {
            aifi,
            encode_proj_layers: cfg.encode_proj_layers.clone(),
            lateral_convs,
            fpn_blocks,
            downsample_convs,
            pan_blocks,
            mask_feature_head: MaskFeatFpn::new(cfg, vb.pp("mask_feature_head"))?,
            encoder_mask_lateral: ConvLayer::new(
                cfg.x4_feat_dim,
                cfg.mask_feature_channels[1],
                3,
                1,
                "silu",
                cfg.batch_norm_eps,
                vb.pp("encoder_mask_lateral"),
            )?,
            encoder_mask_output: EncoderMaskOutput::new(cfg, vb.pp("encoder_mask_output"))?,
        })
    }

    pub fn forward(&self, feature_maps: &mut [Tensor], x4_feat: &Tensor) -> Result<HybridEncoderOutput> {
        if !self.aifi.is_empty() {
            for (i, &enc_ind) in self.encode_proj_layers.iter().enumerate() {
                feature_maps[enc_ind] = self.aifi[i].forward(&feature_maps[enc_ind])?;
            }
        }
        let n_fpn = self.lateral_convs.len();
        let mut fpn_maps = vec![feature_maps[n_fpn].clone()];
        for idx in 0..n_fpn {
            let backbone = &feature_maps[n_fpn - idx - 1];
            let mut top = fpn_maps.last().unwrap().clone();
            top = self.lateral_convs[idx].forward(&top)?;
            *fpn_maps.last_mut().unwrap() = top.clone();
            top = upsample_nearest_2x(&top)?;
            let fused = Tensor::cat(&[&top, backbone], 1)?;
            fpn_maps.push(self.fpn_blocks[idx].forward(&fused)?);
        }
        fpn_maps.reverse();
        let mut pan_maps = vec![fpn_maps[0].clone()];
        for idx in 0..n_fpn {
            let top = pan_maps.last().unwrap();
            let fpn = &fpn_maps[idx + 1];
            let down = self.downsample_convs[idx].forward(top)?;
            let fused = Tensor::cat(&[&down, fpn], 1)?;
            pan_maps.push(self.pan_blocks[idx].forward(&fused)?);
        }
        let mut mask_feat = self.mask_feature_head.forward(&pan_maps)?;
        let (_, _, mh, mw) = mask_feat.dims4()?;
        mask_feat = upsample_bilinear(&mask_feat, mh * 2, mw * 2)?;
        let lat = self.encoder_mask_lateral.forward(x4_feat)?;
        mask_feat = (&mask_feat + &lat)?;
        mask_feat = self.encoder_mask_output.forward(&mask_feat)?;
        Ok(HybridEncoderOutput {
            feature_maps: pan_maps,
            mask_feat,
        })
    }
}

struct EncoderMaskOutput {
    base_conv: ConvLayer,
    conv: Conv2d,
}

impl EncoderMaskOutput {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let ch = cfg.mask_feature_channels[1];
        Ok(Self {
            base_conv: ConvLayer::new(ch, ch, 3, 1, "silu", cfg.batch_norm_eps, vb.pp("base_conv"))?,
            conv: candle_nn::conv2d(ch, cfg.num_prototypes(), 1, Default::default(), vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.base_conv.forward(x)?;
        self.conv.forward(&x)
    }
}

struct MaskFeatFpn {
    scale_heads: Vec<ScaleHead>,
    output_conv: ConvLayer,
    reorder_index: [usize; 3],
}

impl MaskFeatFpn {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let strides = [8usize, 16, 32];
        let mut order: Vec<(usize, usize)> = strides.iter().copied().enumerate().collect();
        order.sort_by_key(|&(_, s)| s);
        let reorder_index = [order[0].0, order[1].0, order[2].0];
        let sorted_strides: Vec<usize> = order.iter().map(|&(_, s)| s).collect();
        let mut scale_heads = Vec::new();
        for i in 0..3 {
            scale_heads.push(ScaleHead::new(
                cfg.encoder_hidden_dim,
                cfg.mask_feature_channels[0],
                sorted_strides[i],
                sorted_strides[0],
                vb.pp(format!("scale_heads.{i}")),
            )?);
        }
        Ok(Self {
            scale_heads,
            output_conv: ConvLayer::new(
                cfg.mask_feature_channels[0],
                cfg.mask_feature_channels[1],
                3,
                1,
                "silu",
                cfg.batch_norm_eps,
                vb.pp("output_conv"),
            )?,
            reorder_index,
        })
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let x: Vec<Tensor> = self.reorder_index.iter().map(|&i| inputs[i].clone()).collect();
        let mut output = self.scale_heads[0].forward(&x[0])?;
        for i in 1..3 {
            let h = output.dims()[2];
            let w = output.dims()[3];
            let head = self.scale_heads[i].forward(&x[i])?;
            let head = upsample_bilinear(&head, h, w)?;
            output = (&output + &head)?;
        }
        self.output_conv.forward(&output)
    }
}

struct ScaleHead {
    layers: Vec<LayerKind>,
}

enum LayerKind {
    Conv(ConvLayer),
    Upsample,
}

impl ScaleHead {
    fn new(in_ch: usize, feat_ch: usize, fpn_stride: usize, base_stride: usize, vb: VarBuilder) -> Result<Self> {
        let head_length = (fpn_stride.ilog2() as i32 - base_stride.ilog2() as i32).max(0) as usize;
        let head_length = head_length.max(1);
        let mut layers = Vec::new();
        let mut layer_idx = 0usize;
        for k in 0..head_length {
            let in_c = if k == 0 { in_ch } else { feat_ch };
            layers.push(LayerKind::Conv(ConvLayer::new(
                in_c,
                feat_ch,
                3,
                1,
                "silu",
                1e-5,
                vb.pp(format!("layers.{layer_idx}")),
            )?));
            layer_idx += 1;
            if fpn_stride != base_stride {
                layers.push(LayerKind::Upsample);
                layer_idx += 1;
            }
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = match layer {
                LayerKind::Conv(c) => c.forward(&x)?,
                LayerKind::Upsample => upsample_bilinear(&x, x.dims()[2] * 2, x.dims()[3] * 2)?,
            };
        }
        Ok(x)
    }
}

struct CspRepLayer {
    conv1: ConvNormLayer,
    conv2: ConvNormLayer,
    bottlenecks: Vec<RepVggBlock>,
    conv3: Option<ConvNormLayer>,
}

impl CspRepLayer {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let in_ch = cfg.encoder_hidden_dim * 2;
        let out_ch = cfg.encoder_hidden_dim;
        let hidden = (out_ch as f64 * cfg.hidden_expansion) as usize;
        let mut bottlenecks = Vec::new();
        for i in 0..3 {
            bottlenecks.push(RepVggBlock::new(cfg, vb.pp(format!("bottlenecks.{i}")))?);
        }
        let conv3 = if hidden != out_ch {
            Some(ConvNormLayer::new(
                cfg, hidden, out_ch, 1, 1, None, Some(&cfg.activation_function), vb.pp("conv3"),
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1: ConvNormLayer::new(
                cfg, in_ch, hidden, 1, 1, None, Some(&cfg.activation_function), vb.pp("conv1"),
            )?,
            conv2: ConvNormLayer::new(
                cfg, in_ch, hidden, 1, 1, None, Some(&cfg.activation_function), vb.pp("conv2"),
            )?,
            bottlenecks,
            conv3,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = self.conv1.forward(x)?;
        let mut h = h1.clone();
        for b in &self.bottlenecks {
            h = b.forward(&h)?;
        }
        let h2 = self.conv2.forward(x)?;
        let h = (&h + &h2)?;
        match &self.conv3 {
            Some(c) => c.forward(&h),
            None => Ok(h),
        }
    }
}

struct RepVggBlock {
    conv1: ConvNormLayer,
    conv2: ConvNormLayer,
    act: String,
}

impl RepVggBlock {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let hidden = (cfg.encoder_hidden_dim as f64 * cfg.hidden_expansion) as usize;
        Ok(Self {
            // Inner convs use Identity activation; silu is applied once on the sum (HF RepVggBlock).
            conv1: ConvNormLayer::new(cfg, hidden, hidden, 3, 1, Some(1), Some("identity"), vb.pp("conv1"))?,
            conv2: ConvNormLayer::new(cfg, hidden, hidden, 1, 1, Some(0), Some("identity"), vb.pp("conv2"))?,
            act: cfg.activation_function.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = (&self.conv1.forward(x)? + &self.conv2.forward(x)?)?;
        activation(&self.act, &y)
    }
}

struct AifiLayer {
    pos_embed: SinePositionEmbedding,
    layers: Vec<EncoderLayer>,
    hidden_dim: usize,
}

impl AifiLayer {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer::new(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            pos_embed: SinePositionEmbedding::new(cfg.encoder_hidden_dim, cfg.positional_encoding_temperature),
            layers,
            hidden_dim: cfg.encoder_hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;
        let mut hs = x.flatten(2, 3)?.transpose(1, 2)?;
        let pos = self.pos_embed.build(h, w, x.device(), x.dtype())?;
        for layer in &self.layers {
            hs = layer.forward(&hs, Some(&pos))?;
        }
        hs.transpose(1, 2)?.reshape((b, self.hidden_dim, h, w))
    }
}

struct SinePositionEmbedding {
    embed_dim: usize,
    temperature: f64,
}

impl SinePositionEmbedding {
    fn new(embed_dim: usize, temperature: f64) -> Self {
        Self {
            embed_dim,
            temperature,
        }
    }

    fn build(&self, height: usize, width: usize, device: &candle_core::Device, dtype: candle_core::DType) -> Result<Tensor> {
        let pos_dim = self.embed_dim / 4;
        let mut omega = Vec::with_capacity(pos_dim);
        for i in 0..pos_dim {
            omega.push(1.0 / self.temperature.powf(i as f64 / pos_dim as f64));
        }
        let mut emb = vec![0f32; height * width * self.embed_dim];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * self.embed_dim;
                for i in 0..pos_dim {
                    let h_angle = y as f64 * omega[i];
                    let w_angle = x as f64 * omega[i];
                    emb[idx + i] = h_angle.sin() as f32;
                    emb[idx + pos_dim + i] = h_angle.cos() as f32;
                    emb[idx + 2 * pos_dim + i] = w_angle.sin() as f32;
                    emb[idx + 3 * pos_dim + i] = w_angle.cos() as f32;
                }
            }
        }
        Tensor::from_vec(emb, (1, height * width, self.embed_dim), device)?.to_dtype(dtype)
    }
}

struct EncoderLayer {
    self_attn: SelfAttention,
    self_attn_ln: LayerNorm,
    mlp: Mlp,
    final_ln: LayerNorm,
    normalize_before: bool,
}

impl EncoderLayer {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.encoder_hidden_dim;
        Ok(Self {
            self_attn: SelfAttention::new(h, cfg.encoder_attention_heads, vb.pp("self_attn"))?,
            self_attn_ln: LayerNorm::new(h, cfg.layer_norm_eps, vb.pp("self_attn_layer_norm"))?,
            mlp: Mlp::new(h, cfg.encoder_ffn_dim, &cfg.encoder_activation_function, vb.clone())?,
            final_ln: LayerNorm::new(h, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?,
            normalize_before: cfg.normalize_before,
        })
    }

    fn forward(&self, hs: &Tensor, pos: Option<&Tensor>) -> Result<Tensor> {
        let mut residual = hs.clone();
        let mut h = if self.normalize_before {
            self.self_attn_ln.forward(hs)?
        } else {
            hs.clone()
        };
        h = self.self_attn.forward(&h, pos)?;
        h = (&residual + &h)?;
        if !self.normalize_before {
            h = self.self_attn_ln.forward(&h)?;
        }
        residual = h.clone();
        h = if self.normalize_before {
            self.final_ln.forward(&h)?
        } else {
            h
        };
        h = self.mlp.forward(&h)?;
        h = (&residual + &h)?;
        if !self.normalize_before {
            h = self.final_ln.forward(&h)?;
        }
        Ok(h)
    }
}

struct SelfAttention {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
    n_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(hidden: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: linear_b(hidden, hidden, vb.pp("q_proj"))?,
            k: linear_b(hidden, hidden, vb.pp("k_proj"))?,
            v: linear_b(hidden, hidden, vb.pp("v_proj"))?,
            o: linear_b(hidden, hidden, vb.pp("out_proj"))?,
            n_heads,
            head_dim: hidden / n_heads,
        })
    }

    fn forward(&self, hs: &Tensor, pos: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, h) = hs.dims3()?;
        let q_in = match pos {
            Some(p) => (hs + p)?,
            None => hs.clone(),
        };
        let q = self.q.forward(&q_in)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k.forward(&q_in)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v.forward(hs)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = candle_nn::ops::softmax_last_dim(&((q.matmul(&k.transpose(2, 3)?)? * scale)?))?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, h))?;
        self.o.forward(&out)
    }
}

pub struct DecoderInputProj {
    layers: Vec<(Conv2d, BatchNorm2d)>,
}

impl DecoderInputProj {
    pub fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..cfg.num_feature_levels {
            let in_ch = if i < cfg.decoder_in_channels.len() {
                cfg.decoder_in_channels[i]
            } else {
                cfg.d_model
            };
            let kernel = if i >= cfg.decoder_in_channels.len() { 3 } else { 1 };
            let stride = if i >= cfg.decoder_in_channels.len() { 2 } else { 1 };
            let cfg_conv = Conv2dConfig {
                stride,
                padding: kernel / 2,
                ..Default::default()
            };
            let conv = candle_nn::conv2d_no_bias(in_ch, cfg.d_model, kernel, cfg_conv, vb.pp(format!("{i}.0")))?;
            let norm = BatchNorm2d::load(cfg.d_model, cfg.batch_norm_eps, vb.pp(format!("{i}.1")))?;
            layers.push((conv, norm));
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, i: usize) -> Result<Tensor> {
        let x = self.layers[i].0.forward(x)?;
        self.layers[i].1.forward(&x)
    }
}
