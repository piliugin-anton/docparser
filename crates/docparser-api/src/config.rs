use std::path::PathBuf;

use docparser_pipeline::PipelineConfig;

/// Load `.env` from the current working directory when present.
/// Existing environment variables are not overridden.
pub fn load_env_file() {
    let _ = dotenvy::dotenv();
}

pub struct ApiConfig {
    pub bind_addr: String,
    pub models_dir: PathBuf,
    pub max_upload_mb: usize,
    pub max_tokens: usize,
    pub pipeline: PipelineConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        load_env_file();
        let official = std::env::var("PIPELINE_PROFILE")
            .map(|v| v.eq_ignore_ascii_case("official_v16") || v == "v1.6")
            .unwrap_or(false);
        let mut pipeline = if official {
            PipelineConfig::official_v16()
        } else {
            PipelineConfig::minimal()
        };
        pipeline.max_tokens = std::env::var("MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        if let Ok(v) = std::env::var("LAYOUT_UNCLIP_RATIO") {
            if let Ok(f) = v.parse() {
                pipeline.layout_unclip_ratio = f;
            }
        }
        if let Ok(v) = std::env::var("LAYOUT_THRESHOLD") {
            if let Ok(f) = v.parse() {
                pipeline.layout_threshold = f;
            }
        }
        if let Ok(v) = std::env::var("LAYOUT_NMS") {
            pipeline.layout_nms = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("MERGE_LAYOUT_BLOCKS") {
            pipeline.merge_layout_blocks = v == "1" || v.eq_ignore_ascii_case("true");
        }
        Self {
            bind_addr: std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into()),
            models_dir: std::env::var("MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("models")),
            max_upload_mb: std::env::var("MAX_UPLOAD_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            max_tokens: pipeline.max_tokens,
            pipeline,
        }
    }
}
