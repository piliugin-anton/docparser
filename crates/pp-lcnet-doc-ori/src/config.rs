use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::{DocOriError, Result};

#[derive(Debug, Clone)]
pub struct PpLcnetConfig {
    pub scale: f64,
    pub reduction: usize,
    pub hidden_dropout_prob: f32,
    pub class_expand: usize,
    pub hidden_act: String,
    pub stem_channels: usize,
    pub stem_stride: usize,
    pub divisor: usize,
    pub block_configs: Vec<Vec<BlockSpec>>,
    pub id2label: HashMap<u32, String>,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockSpec {
    pub kernel: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
    pub use_se: bool,
}

impl PpLcnetConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&path)
            .map_err(|e| DocOriError::Message(format!("read {}: {e}", path.display())))?;
        #[derive(Deserialize)]
        struct Root {
            scale: Option<f64>,
            reduction: Option<usize>,
            hidden_dropout_prob: Option<f32>,
            class_expand: Option<usize>,
            hidden_act: Option<String>,
            id2label: serde_json::Map<String, serde_json::Value>,
        }
        let root: Root = serde_json::from_str(&data)?;
        let id2label: HashMap<u32, String> = root
            .id2label
            .iter()
            .filter_map(|(k, v)| {
                let id: u32 = k.parse().ok()?;
                Some((id, v.as_str()?.to_string()))
            })
            .collect();
        Ok(Self {
            scale: root.scale.unwrap_or(1.0),
            reduction: root.reduction.unwrap_or(4),
            hidden_dropout_prob: root.hidden_dropout_prob.unwrap_or(0.2),
            class_expand: root.class_expand.unwrap_or(1280),
            hidden_act: root.hidden_act.unwrap_or_else(|| "hardswish".into()),
            stem_channels: 16,
            stem_stride: 2,
            divisor: 8,
            block_configs: default_block_configs(),
            id2label,
        })
    }

    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    pub fn angle_for_class(&self, class_id: usize) -> u32 {
        self.id2label
            .get(&(class_id as u32))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

fn default_block_configs() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![BlockSpec {
            kernel: 3,
            in_channels: 16,
            out_channels: 32,
            stride: 1,
            use_se: false,
        }],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 32,
                out_channels: 64,
                stride: 2,
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 64,
                out_channels: 64,
                stride: 1,
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 64,
                out_channels: 128,
                stride: 2,
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 128,
                out_channels: 128,
                stride: 1,
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 128,
                out_channels: 256,
                stride: 2,
                use_se: false,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 256,
                stride: 1,
                use_se: false,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 256,
                stride: 1,
                use_se: false,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 256,
                stride: 1,
                use_se: false,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 256,
                stride: 1,
                use_se: false,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 256,
                stride: 1,
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 5,
                in_channels: 256,
                out_channels: 512,
                stride: 2,
                use_se: true,
            },
            BlockSpec {
                kernel: 5,
                in_channels: 512,
                out_channels: 512,
                stride: 1,
                use_se: true,
            },
        ],
    ]
}

pub fn make_divisible(v: f64, divisor: usize) -> usize {
    let divisor = divisor.max(1);
    let min_value = divisor;
    let mut new_v = ((v + divisor as f64 / 2.0) / divisor as f64).floor() as usize * divisor;
    new_v = new_v.max(min_value);
    if (new_v as f64) < 0.9 * v {
        new_v += divisor;
    }
    new_v
}
