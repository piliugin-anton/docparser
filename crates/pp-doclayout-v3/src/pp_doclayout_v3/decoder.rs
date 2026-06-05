//! PP-DocLayoutV3 transformer decoder.

use candle_core::{Result, Tensor};
use candle_nn::{ops::sigmoid, Linear, Module, VarBuilder};

use super::config::PpDocLayoutV3Config;
use super::deformable::MultiscaleDeformableAttention;
use super::global_pointer::GlobalPointer;
use super::nn::{linear_b, LayerNorm, Mlp, MlpPredictionHead};

pub struct DecoderOutput {
    pub intermediate_logits: Tensor,
    pub intermediate_reference_points: Tensor,
    pub decoder_out_order_logits: Tensor,
    pub decoder_out_masks: Tensor,
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    query_pos_head: MlpPredictionHead,
}

impl Decoder {
    pub fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..cfg.decoder_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            layers,
            query_pos_head: MlpPredictionHead::new(4, 2 * cfg.d_model, cfg.d_model, 2, vb.pp("query_pos_head"))?,
        })
    }

    // DETR-style decoder step mirrors upstream API (embeds, refs, heads, mask feat).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        encoder_hidden_states: &Tensor,
        reference_points: &Tensor,
        spatial_shapes: &[(usize, usize)],
        bbox_embed: &MlpPredictionHead,
        class_embed: &Linear,
        order_heads: &[Linear],
        global_pointer: &GlobalPointer,
        mask_query_head: &MlpPredictionHead,
        norm: &LayerNorm,
        mask_feat: &Tensor,
        num_queries: usize,
    ) -> Result<DecoderOutput> {
        let mut hidden_states = inputs_embeds.clone();
        let mut reference_points = sigmoid(reference_points)?;
        let batch = hidden_states.dims()[0];
        let mut inter_refs = Vec::new();
        let mut inter_logits = Vec::new();
        let mut order_logits_list = Vec::new();
        let mut masks_list = Vec::new();

        for (idx, layer) in self.layers.iter().enumerate() {
            let ref_in = reference_points.unsqueeze(2)?;
            let pos = self.query_pos_head.forward(&reference_points)?;
            hidden_states = layer.forward(
                &hidden_states,
                Some(&pos),
                &ref_in,
                encoder_hidden_states,
                spatial_shapes,
            )?;
            let predicted = bbox_embed.forward(&hidden_states)?;
            let new_ref = sigmoid(&(predicted + inverse_sigmoid(&reference_points)?)?)?;
            reference_points = new_ref;
            inter_refs.push(reference_points.clone());

            let out_query = norm.forward(&hidden_states)?;
            let mask_query_embed = mask_query_head.forward(&out_query)?;
            let (_b, mask_dim, _) = mask_query_embed.dims3()?;
            let (_b2, _np, mask_h, mask_w) = mask_feat.dims4()?;
            let mf = mask_feat.flatten(2, 3)?;
            let out_mask = docparser_candle_utils::matmul(&mask_query_embed, &mf)?
                .reshape((batch, mask_dim, mask_h, mask_w))?;
            masks_list.push(out_mask);

            inter_logits.push(class_embed.forward(&out_query)?);

            let valid = out_query.narrow(1, out_query.dims()[1].saturating_sub(num_queries), num_queries)?;
            let order_in = order_heads[idx].forward(&valid)?;
            order_logits_list.push(global_pointer.forward(&order_in)?);
        }

        Ok(DecoderOutput {
            intermediate_logits: Tensor::stack(&inter_logits, 1)?,
            intermediate_reference_points: Tensor::stack(&inter_refs, 1)?,
            decoder_out_order_logits: Tensor::stack(&order_logits_list, 1)?,
            decoder_out_masks: Tensor::stack(&masks_list, 1)?,
        })
    }
}

struct DecoderLayer {
    self_attn: DecoderSelfAttention,
    self_attn_ln: LayerNorm,
    cross_attn: MultiscaleDeformableAttention,
    cross_attn_ln: LayerNorm,
    mlp: Mlp,
    final_ln: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &PpDocLayoutV3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.d_model;
        Ok(Self {
            self_attn: DecoderSelfAttention::new(h, cfg.decoder_attention_heads, vb.pp("self_attn"))?,
            self_attn_ln: LayerNorm::new(h, cfg.layer_norm_eps, vb.pp("self_attn_layer_norm"))?,
            cross_attn: MultiscaleDeformableAttention::new(
                cfg,
                cfg.decoder_attention_heads,
                cfg.decoder_n_points,
                vb.pp("encoder_attn"),
            )?,
            cross_attn_ln: LayerNorm::new(h, cfg.layer_norm_eps, vb.pp("encoder_attn_layer_norm"))?,
            mlp: Mlp::new(h, cfg.decoder_ffn_dim, &cfg.decoder_activation_function, vb.clone())?,
            final_ln: LayerNorm::new(h, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &self,
        hs: &Tensor,
        pos: Option<&Tensor>,
        ref_points: &Tensor,
        enc: &Tensor,
        spatial_shapes: &[(usize, usize)],
    ) -> Result<Tensor> {
        let residual = hs.clone();
        let mut h = self.self_attn.forward(hs, pos)?;
        h = (&residual + &h)?;
        h = self.self_attn_ln.forward(&h)?;
        let residual = h.clone();
        h = self.cross_attn.forward(&h, enc, pos, ref_points, spatial_shapes)?;
        h = (&residual + &h)?;
        h = self.cross_attn_ln.forward(&h)?;
        let residual = h.clone();
        h = self.mlp.forward(&h)?;
        (&residual + &h).and_then(|x| self.final_ln.forward(&x))
    }
}

struct DecoderSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl DecoderSelfAttention {
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
        let attn = candle_nn::ops::softmax_last_dim(&((docparser_candle_utils::matmul_transpose(&q, &k, 2, 3)? * scale)?))?;
        let out = docparser_candle_utils::matmul_contig_rhs(&attn, &v)?
            .transpose(1, 2)?
            .reshape((b, s, h))?;
        self.o.forward(&out)
    }
}

pub fn inverse_sigmoid(x: &Tensor) -> Result<Tensor> {
    let x = x.clamp(0.0, 1.0)?;
    let eps = 1e-5f64;
    let x1 = x.clamp(eps, 1.0)?;
    let x2 = ((&Tensor::ones_like(&x)? - &x)?).clamp(eps, 1.0)?;
    (x1 / x2)?.log()
}
