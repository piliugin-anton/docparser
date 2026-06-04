use std::path::PathBuf;

use docparser_pipeline::{DocPreprocessorConfig, PipelineConfig};

/// Load `.env` from the current working directory when present.
/// Existing environment variables are not overridden.
pub fn load_env_file() {
    let _ = dotenvy::dotenv();
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
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
        let mut pipeline = PipelineConfig::default();
        pipeline.max_tokens = std::env::var("MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        if let Ok(v) = std::env::var("LAYOUT_UNCLIP_RATIO") {
            if let Ok(f) = v.parse() {
                pipeline.layout_unclip_ratio = f;
            }
        }
        if let Ok(v) = std::env::var("CROP_PADDING_RATIO") {
            if let Ok(f) = v.parse() {
                pipeline.crop_padding_ratio = f;
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
        pipeline.doc_preprocess = DocPreprocessorConfig {
            use_orientation: env_bool("USE_DOC_ORIENTATION_CLASSIFY", true),
            use_unwarping: env_bool("USE_DOC_UNWARPING", true),
        };
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
