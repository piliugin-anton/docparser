use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("VLM error: {0}")]
    Vlm(#[from] paddleocr_vl::VlmError),
    #[error("layout error: {0}")]
    Layout(#[from] pp_doclayout_v3::LayoutError),
    #[error("doc orientation error: {0}")]
    DocOri(#[from] pp_lcnet_doc_ori::DocOriError),
    #[error("UVDoc error: {0}")]
    Uvdoc(#[from] uvdoc::UvdocError),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Message(String),
}

impl From<anyhow::Error> for PipelineError {
    fn from(err: anyhow::Error) -> Self {
        Self::Message(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, PipelineError>;
