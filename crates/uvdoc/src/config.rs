use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::{Result, UvdocError};

fn json_u64(value: &Value, field: &str) -> Result<usize> {
    value
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| UvdocError::InvalidConfigField {
            field: field.to_string(),
            value: value.to_string(),
        })
}

#[derive(Debug, Clone)]
pub struct UvdocBackboneConfig {
    pub resnet_head: Vec<[usize; 2]>,
    pub resnet_configs: Vec<Vec<[usize; 4]>>,
    pub stage_configs: Vec<Vec<[usize; 2]>>,
    pub kernel_size: usize,
}

#[derive(Debug, Clone)]
pub struct UvdocConfig {
    pub kernel_size: usize,
    pub bridge_connector: [usize; 2],
    pub out_point_positions2d: [[usize; 2]; 2],
    pub padding_mode: String,
    pub hidden_act: String,
    pub upsample_size: [usize; 2],
    pub backbone: UvdocBackboneConfig,
}

impl UvdocConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let data = std::fs::read_to_string(&path)
            .map_err(|e| UvdocError::Message(format!("read {}: {e}", path.display())))?;
        #[derive(Deserialize)]
        struct Root {
            kernel_size: Option<usize>,
            bridge_connector: Option<Vec<usize>>,
            #[serde(rename = "out_point_positions2D")]
            out_point_positions2d: Option<Vec<Vec<usize>>>,
            padding_mode: Option<String>,
            hidden_act: Option<String>,
            backbone_config: BackboneRoot,
        }
        #[derive(Deserialize)]
        struct BackboneRoot {
            resnet_head: Vec<Vec<usize>>,
            resnet_configs: Vec<Vec<Vec<serde_json::Value>>>,
            stage_configs: Vec<Vec<Vec<usize>>>,
            kernel_size: Option<usize>,
        }
        let root: Root = serde_json::from_str(&data)?;
        let resnet_head = root
            .backbone_config
            .resnet_head
            .iter()
            .map(|p| [p[0], p[1]])
            .collect();
        let mut resnet_configs = Vec::new();
        for stage in &root.backbone_config.resnet_configs {
            let mut specs = Vec::new();
            for item in stage {
                if item.len() < 4 {
                    return Err(UvdocError::InvalidResnetConfig);
                }
                let in_ch = json_u64(&item[0], "resnet_configs.in_ch")?;
                let out_ch = json_u64(&item[1], "resnet_configs.out_ch")?;
                let dilation = json_u64(&item[2], "resnet_configs.dilation")?;
                let downsample = item[3].as_bool().unwrap_or(false);
                specs.push([in_ch, out_ch, dilation, downsample as usize]);
            }
            resnet_configs.push(specs);
        }
        let stage_configs = root
            .backbone_config
            .stage_configs
            .iter()
            .map(|stage| stage.iter().map(|p| [p[0], p[1]]).collect())
            .collect();
        let out_point = root
            .out_point_positions2d
            .unwrap_or_else(|| vec![vec![128, 32], vec![32, 2]]);
        Ok(Self {
            kernel_size: root.kernel_size.unwrap_or(5),
            bridge_connector: {
                let v = root.bridge_connector.unwrap_or_else(|| vec![128, 128]);
                [v[0], v[1]]
            },
            out_point_positions2d: [
                [out_point[0][0], out_point[0][1]],
                [out_point[1][0], out_point[1][1]],
            ],
            padding_mode: root.padding_mode.unwrap_or_else(|| "reflect".into()),
            hidden_act: root.hidden_act.unwrap_or_else(|| "prelu".into()),
            upsample_size: [712, 488],
            backbone: UvdocBackboneConfig {
                resnet_head,
                resnet_configs,
                stage_configs,
                kernel_size: root.backbone_config.kernel_size.unwrap_or(5),
            },
        })
    }
}
