use std::path::PathBuf;

pub struct ApiConfig {
    pub bind_addr: String,
    pub models_dir: PathBuf,
    pub max_upload_mb: usize,
    pub max_tokens: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind_addr: std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into()),
            models_dir: std::env::var("MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("models")),
            max_upload_mb: std::env::var("MAX_UPLOAD_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            max_tokens: std::env::var("MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
        }
    }
}
