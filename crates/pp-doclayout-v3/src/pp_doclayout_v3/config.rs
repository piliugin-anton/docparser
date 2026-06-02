//! PP-DocLayoutV3 configuration (from `config.json`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct BackboneConfig {
    pub arch: String,
    #[serde(default)]
    pub return_idx: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PpDocLayoutV3Config {
    pub d_model: usize,
    pub num_queries: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_n_points: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_hidden_dim: usize,
    pub encoder_in_channels: Vec<usize>,
    pub decoder_in_channels: Vec<usize>,
    pub encode_proj_layers: Vec<usize>,
    pub feature_strides: Vec<usize>,
    pub num_feature_levels: usize,
    pub layer_norm_eps: f64,
    pub batch_norm_eps: f64,
    pub dropout: f64,
    pub attention_dropout: f64,
    pub activation_dropout: f64,
    pub activation_function: String,
    pub encoder_activation_function: String,
    pub decoder_activation_function: String,
    pub positional_encoding_temperature: f64,
    pub hidden_expansion: f64,
    pub normalize_before: bool,
    pub num_denoising: usize,
    pub learn_initial_query: bool,
    pub global_pointer_head_size: usize,
    pub mask_feature_channels: Vec<usize>,
    pub x4_feat_dim: usize,
    pub backbone_config: BackboneConfig,
    #[serde(default)]
    pub mask_enhanced: bool,
    #[serde(default = "default_gp_dropout")]
    pub gp_dropout_value: f64,
    #[serde(default = "default_true")]
    pub freeze_backbone_batch_norms: bool,
    pub id2label: serde_json::Map<String, serde_json::Value>,
}

fn default_gp_dropout() -> f64 {
    0.1
}

fn default_true() -> bool {
    true
}

impl PpDocLayoutV3Config {
    pub fn from_dir(model_dir: &std::path::Path) -> anyhow::Result<Self> {
        let path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    pub fn num_prototypes(&self) -> usize {
        32
    }

    pub fn label_for_id(&self, id: i64) -> String {
        self.id2label
            .get(&id.to_string())
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub fn id2label_map(&self) -> std::collections::HashMap<u32, String> {
        self.id2label
            .iter()
            .filter_map(|(k, v)| {
                let id: u32 = k.parse().ok()?;
                let name = v.as_str()?.to_string();
                Some((id, name))
            })
            .collect()
    }

    pub fn hgnet(&self) -> HgNetV2Config {
        HgNetV2Config::arch_l()
    }
}

/// HGNet-V2 arch L (matches Transformers `HGNetV2Config` for PP-DocLayoutV3).
#[derive(Debug, Clone)]
pub struct HgNetV2Config {
    pub num_channels: usize,
    pub stem_channels: [usize; 3],
    pub stem_strides: [usize; 5],
    pub stage_in_channels: [usize; 4],
    pub stage_mid_channels: [usize; 4],
    pub stage_out_channels: [usize; 4],
    pub stage_num_blocks: [usize; 4],
    pub stage_numb_of_layers: [usize; 4],
    pub stage_downsample: [bool; 4],
    pub stage_light_block: [bool; 4],
    pub stage_kernel_size: [usize; 4],
    pub stage_downsample_strides: [usize; 4],
    pub hidden_act: String,
    pub use_learnable_affine_block: bool,
}

impl HgNetV2Config {
    pub fn arch_l() -> Self {
        Self {
            num_channels: 3,
            stem_channels: [3, 32, 48],
            stem_strides: [2, 1, 1, 2, 1],
            stage_in_channels: [48, 128, 512, 1024],
            stage_mid_channels: [48, 96, 192, 384],
            stage_out_channels: [128, 512, 1024, 2048],
            stage_num_blocks: [1, 1, 3, 1],
            stage_numb_of_layers: [6, 6, 6, 6],
            stage_downsample: [false, true, true, true],
            stage_light_block: [false, false, true, true],
            stage_kernel_size: [3, 3, 5, 5],
            stage_downsample_strides: [2, 2, 2, 2],
            hidden_act: "relu".to_string(),
            use_learnable_affine_block: false,
        }
    }

    pub fn channels(&self) -> [usize; 4] {
        self.stage_out_channels
    }
}
