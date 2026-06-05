use thiserror::Error;

use docparser_candle_utils::CandleUtilsError;

#[derive(Debug, Error)]
pub enum VlmError {
    #[error("candle utils: {0}")]
    CandleUtils(#[from] CandleUtilsError),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("runner lock poisoned")]
    LockPoisoned,
    #[error("runner missing after initialization")]
    RunnerNotLoaded,
    #[error("aspect ratio {aspect} exceeds maximum 200")]
    InvalidAspectRatio { aspect: f64 },
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, VlmError>;
