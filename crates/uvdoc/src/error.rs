use thiserror::Error;

use docparser_candle_utils::CandleUtilsError;

#[derive(Debug, Error)]
pub enum UvdocError {
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
    #[error("config field `{field}`: expected unsigned integer, got {value}")]
    InvalidConfigField { field: String, value: String },
    #[error("resnet_configs entry must have at least 4 fields")]
    InvalidResnetConfig,
    #[error("expected 3-channel output, got {channels}")]
    InvalidChannelCount { channels: usize },
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, UvdocError>;
