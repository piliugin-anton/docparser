//! PP-DocLayoutV3 full model forward (inference).

use candle_core::{Device, DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;
use super::decoder::{inverse_sigmoid, Decoder};
use super::encoder::{ConvEncoder, DecoderInputProj, EncoderInputProj, HybridEncoder};
use super::global_pointer::GlobalPointer;
use super::nn::{LayerNorm, MlpPredictionHead};
use super::ops::{gather_dim, topk_last_dim};

pub struct ModelOutputs {
    pub logits: Tensor,
    pub pred_boxes: Tensor,
    pub order_logits: Tensor,
    pub out_masks: Tensor,
}

pub struct PpDocLayoutV3Model {
    backbone: ConvEncoder,
    encoder_input_proj: EncoderInputProj,
    encoder: HybridEncoder,
    decoder_input_proj: DecoderInputProj,
    decoder: Decoder,
    enc_output: (Linear, LayerNorm),
    enc_score_head: Linear,
    enc_bbox_head: MlpPredictionHead,
    decoder_order_head: Vec<Linear>,
    decoder_global_pointer: GlobalPointer,
    decoder_norm: LayerNorm,
    mask_query_head: MlpPredictionHead,
    mask_enhanced: bool,
    num_queries: usize,
}

impl PpDocLayoutV3Model {
    pub fn load(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let channel_sizes = [512usize, 1024, 2048];
        let mut order_heads = Vec::new();
        for i in 0..cfg.decoder_layers {
            order_heads.push(candle_nn::linear(
                cfg.d_model,
                cfg.d_model,
                vb.pp(format!("decoder_order_head.{i}")),
            )?);
        }
        Ok(Self {
            backbone: ConvEncoder::new(cfg, vb.pp("backbone"))?,
            encoder_input_proj: EncoderInputProj::new(cfg, &channel_sizes, vb.pp("encoder_input_proj"))?,
            encoder: HybridEncoder::new(cfg, vb.pp("encoder"))?,
            decoder_input_proj: DecoderInputProj::new(cfg, vb.pp("decoder_input_proj"))?,
            decoder: Decoder::new(cfg, vb.pp("decoder"))?,
            enc_output: (
                candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("enc_output.0"))?,
                LayerNorm::new(cfg.d_model, cfg.layer_norm_eps, vb.pp("enc_output.1"))?,
            ),
            enc_score_head: candle_nn::linear(cfg.d_model, cfg.num_labels(), vb.pp("enc_score_head"))?,
            enc_bbox_head: MlpPredictionHead::new(cfg.d_model, cfg.d_model, 4, 3, vb.pp("enc_bbox_head"))?,
            decoder_order_head: order_heads,
            decoder_global_pointer: GlobalPointer::new(cfg, vb.pp("decoder_global_pointer"))?,
            decoder_norm: LayerNorm::new(cfg.d_model, cfg.layer_norm_eps, vb.pp("decoder_norm"))?,
            mask_query_head: MlpPredictionHead::new(
                cfg.d_model,
                cfg.d_model,
                cfg.num_prototypes(),
                3,
                vb.pp("mask_query_head"),
            )?,
            mask_enhanced: cfg.mask_enhanced,
            num_queries: cfg.num_queries,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, pixel_mask: &Tensor) -> Result<ModelOutputs> {
        let (x4_feat, mut features) = self.backbone.forward(pixel_values, pixel_mask)?;
        let x4 = x4_feat.clone();
        features.remove(0);
        let mut proj_feats = Vec::new();
        for (i, (feat, _mask)) in features.into_iter().enumerate() {
            proj_feats.push(self.encoder_input_proj.forward(&feat, i)?);
        }
        let enc_out = self.encoder.forward(&mut proj_feats, &x4)?;

        let mut sources = Vec::new();
        for (level, feat) in enc_out.feature_maps.iter().enumerate() {
            sources.push(self.decoder_input_proj.forward(feat, level)?);
        }
        let len_src = sources.len();
        let num_levels = 3usize;
        if num_levels > len_src {
            let last = enc_out.feature_maps.last().unwrap();
            sources.push(self.decoder_input_proj.forward(last, len_src)?);
            for i in len_src + 1..num_levels {
                sources.push(self.decoder_input_proj.forward(last, i)?);
            }
        }

        let mut spatial_shapes = Vec::new();
        let mut flat = Vec::new();
        for src in &sources {
            let (_b, _c, h, w) = src.dims4()?;
            spatial_shapes.push((h, w));
            flat.push(src.flatten(2, 3)?.transpose(1, 2)?);
        }
        let source_flatten = Tensor::cat(&flat, 1)?;
        let (anchors, valid_mask) =
            generate_anchors(&spatial_shapes, source_flatten.device(), source_flatten.dtype())?;
        let memory = source_flatten.broadcast_mul(&valid_mask.to_dtype(source_flatten.dtype())?)?;
        let output_memory = self.enc_output.1.forward(&self.enc_output.0.forward(&memory)?)?;
        let enc_outputs_class = self.enc_score_head.forward(&output_memory)?;
        let enc_outputs_coord = (&self.enc_bbox_head.forward(&output_memory)? + &anchors)?;

        let max_cls = enc_outputs_class.max(D::Minus1)?;
        let (_top_scores, topk_ind) = topk_last_dim(&max_cls, self.num_queries)?;

        let reference_points_unact = gather_dim(&enc_outputs_coord, &topk_ind, 1)?;
        let target = gather_dim(&output_memory, &topk_ind, 1)?.detach();
        let out_query = self.decoder_norm.forward(&target)?;
        let mask_query_embed = self.mask_query_head.forward(&out_query)?;

        let mut reference_points_unact = reference_points_unact;
        if self.mask_enhanced {
            let enc_out_masks = mask_logits_to_boxes(&mask_query_embed, &enc_out.mask_feat)?;
            reference_points_unact = inverse_sigmoid(&enc_out_masks)?;
        }

        let dec = self.decoder.forward(
            &target,
            &source_flatten,
            &reference_points_unact,
            &spatial_shapes,
            &self.enc_bbox_head,
            &self.enc_score_head,
            &self.decoder_order_head,
            &self.decoder_global_pointer,
            &self.mask_query_head,
            &self.decoder_norm,
            &enc_out.mask_feat,
            self.num_queries,
        )?;

        let n_layers = dec.intermediate_logits.dims()[1];
        let logits = dec.intermediate_logits.narrow(1, n_layers - 1, 1)?.squeeze(1)?;
        let pred_boxes = dec
            .intermediate_reference_points
            .narrow(1, n_layers - 1, 1)?
            .squeeze(1)?;
        let order_logits = dec.decoder_out_order_logits.narrow(1, n_layers - 1, 1)?.squeeze(1)?;
        let out_masks = dec.decoder_out_masks.narrow(1, n_layers - 1, 1)?.squeeze(1)?;

        Ok(ModelOutputs {
            logits,
            pred_boxes,
            order_logits,
            out_masks,
        })
    }
}

fn generate_anchors(
    spatial_shapes: &[(usize, usize)],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let grid_size = 0.05f32;
    let eps = 1e-2f32;
    let max_logit = f32::MAX;
    let total: usize = spatial_shapes.iter().map(|(h, w)| h * w).sum();
    let mut all = Vec::with_capacity(total * 4);
    let mut valid = Vec::with_capacity(total);
    for (level, &(height, width)) in spatial_shapes.iter().enumerate() {
        let wh = grid_size * 2f32.powi(level as i32);
        for y in 0..height {
            for x in 0..width {
                let cx = (x as f32 + 0.5) / width as f32;
                let cy = (y as f32 + 0.5) / height as f32;
                let ok = cx > eps && cy > eps && cx < 1.0 - eps && cy < 1.0 - eps;
                if ok {
                    all.push((cx / (1.0 - cx)).ln());
                    all.push((cy / (1.0 - cy)).ln());
                    all.push((wh / (1.0 - wh)).ln());
                    all.push((wh / (1.0 - wh)).ln());
                    valid.push(1.0);
                } else {
                    all.extend([max_logit; 4]);
                    valid.push(0.0);
                }
            }
        }
    }
    let anchors = Tensor::from_vec(all, (1, total, 4), device)?.to_dtype(dtype)?;
    let valid_mask = Tensor::from_vec(valid, (1, total, 1), device)?.to_dtype(dtype)?;
    Ok((anchors, valid_mask))
}

fn mask_logits_to_boxes(mask_query_embed: &Tensor, mask_feat: &Tensor) -> Result<Tensor> {
    let (batch, _mask_dim, _) = mask_query_embed.dims3()?;
    let (_b, _np, mask_h, mask_w) = mask_feat.dims4()?;
    let mf = mask_feat.flatten(2, 3)?;
    let masks = docparser_candle_utils::matmul(mask_query_embed, &mf)?
        .reshape((batch, (), mask_h, mask_w))?;
    let masks = masks.gt(0.0)?.to_dtype(mask_query_embed.dtype())?;
    mask_to_box_coordinate(&masks)
}

fn mask_to_box_coordinate(mask: &Tensor) -> Result<Tensor> {
    let (batch, num_q, h, w) = mask.dims4()?;
    let data = mask.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; batch * num_q * 4];
    for b in 0..batch {
        for q in 0..num_q {
            let base = ((b * num_q + q) * h * w) as usize;
            let mut x_min = w as f32;
            let mut y_min = h as f32;
            let mut x_max = 0f32;
            let mut y_max = 0f32;
            let mut any = false;
            for y in 0..h {
                for x in 0..w {
                    if data[base + y * w + x] > 0.5 {
                        any = true;
                        x_min = x_min.min(x as f32);
                        y_min = y_min.min(y as f32);
                        x_max = x_max.max(x as f32 + 1.0);
                        y_max = y_max.max(y as f32 + 1.0);
                    }
                }
            }
            let o = (b * num_q + q) * 4;
            if any {
                out[o] = (x_min + x_max) / 2.0 / w as f32;
                out[o + 1] = (y_min + y_max) / 2.0 / h as f32;
                out[o + 2] = (x_max - x_min) / w as f32;
                out[o + 3] = (y_max - y_min) / h as f32;
            }
        }
    }
    Tensor::from_vec(out, (batch, num_q, 4), mask.device())
}

