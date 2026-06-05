use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing {kind} artifact: {path}")]
    MissingArtifact { kind: &'static str, path: String },
}

pub type VerifyResult<T> = std::result::Result<T, VerifyError>;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("download task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("download semaphore closed")]
    SemaphoreClosed,
    #[error("verify error: {0}")]
    Verify(#[from] VerifyError),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, DownloadError>;
