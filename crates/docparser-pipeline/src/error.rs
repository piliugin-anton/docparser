use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("VLM error: {0}")]
    Vlm(#[from] paddleocr_vl::VlmError),
    #[error("layout error: {0}")]
    Layout(#[from] pp_doclayout_v3::LayoutError),
    #[error("doc orientation error: {0}")]
    DocOri(#[from] docparser_doc_prep::orientation::DocOriError),
    #[error("UVDoc error: {0}")]
    Uvdoc(#[from] docparser_doc_prep::unwarp::UvdocError),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("doc orientation model path required when use_orientation=true")]
    MissingDocOriPath,
    #[error("UVDoc model path required when use_unwarping=true")]
    MissingUvdocPath,
    #[error("inference queue is full; retry later")]
    InferenceQueueFull,
    #[error("inference worker is unavailable")]
    InferenceWorkerUnavailable,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;
