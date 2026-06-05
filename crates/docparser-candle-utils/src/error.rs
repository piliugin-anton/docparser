use thiserror::Error;

#[derive(Debug, Error)]
pub enum CandleUtilsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, CandleUtilsError>;
