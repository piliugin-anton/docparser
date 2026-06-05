use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use docparser_download::verify_models_dir;
use docparser_pipeline::DocumentPipeline;
use image::ImageFormat;
use serde::Serialize;
use tokio::net::TcpListener;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::info;

mod config;
mod inference;

pub use config::{ApiConfig, load_env_file};
pub use inference::InferencePool;

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<InferencePool>,
}

impl AppState {
    pub fn new(pipeline: DocumentPipeline, queue_depth: usize) -> Self {
        Self {
            inference: Arc::new(InferencePool::new(pipeline, queue_depth)),
        }
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    models_loaded: bool,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

pub fn build_router(state: AppState, max_upload_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/parse", post(parse_document))
        .layer(RequestBodyLimitLayer::new(max_upload_bytes))
        .with_state(state)
}

pub async fn run(config: ApiConfig) -> Result<()> {
    let models_dir = config.models_dir.clone();
    let mut pipeline_cfg = config.pipeline.clone();
    pipeline_cfg.max_tokens = config.max_tokens;

    let pipeline = tokio::task::spawn_blocking(move || {
        verify_models_dir(&models_dir)
            .context("model artifacts missing; run: cargo run -p docparser-download")?;
        DocumentPipeline::from_models_dir(&models_dir, pipeline_cfg)
            .context("failed to load inference models")
    })
    .await
    .context("model load task panicked")??;

    info!("models loaded from {}", config.models_dir.display());

    let state = AppState::new(pipeline, config.inference_queue_depth);
    let app = build_router(state, config.max_upload_mb * 1024 * 1024);
    let addr: SocketAddr = config
        .bind_addr
        .parse()
        .context("invalid DOCPARSER_BIND_ADDR")?;
    info!("listening on http://{addr}");
    let listener = TcpListener::bind(addr)
        .await
        .context("failed to bind HTTP listener")?;
    axum::serve(listener, app)
        .await
        .context("HTTP server exited with error")?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        models_loaded: Arc::strong_count(&state.inference) > 0,
    })
}

async fn parse_document(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, AppError> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("multipart error: {e}")))?
    {
        if field.name() == Some("file") {
            filename = field.file_name().map(str::to_string);
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::bad_request(format!("read upload: {e}")))?;
            file_bytes = Some(data.to_vec());
        }
    }

    let bytes = file_bytes.ok_or_else(|| AppError::bad_request("missing file field"))?;
    let format = detect_image_format(&bytes, filename.as_deref())
        .ok_or_else(|| AppError::unsupported("unsupported image format; use jpg/jpeg/png"))?;

    let result = state
        .inference
        .parse_image(bytes, format, filename)
        .await
        .map_err(pipeline_error_to_app_error)?;

    Ok((StatusCode::OK, Json(result)))
}

fn pipeline_error_to_app_error(err: docparser_pipeline::PipelineError) -> AppError {
    use docparser_pipeline::PipelineError;

    match err {
        PipelineError::Image(_) => AppError::bad_request("invalid or corrupt image data"),
        PipelineError::InferenceQueueFull => {
            AppError::service_unavailable("inference queue is full; retry later")
        }
        PipelineError::InferenceWorkerUnavailable => {
            AppError::service_unavailable("inference worker is unavailable")
        }
        other => AppError::internal(other.to_string()),
    }
}

fn detect_image_format(bytes: &[u8], filename: Option<&str>) -> Option<ImageFormat> {
    if let Some(name) = filename {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            return Some(ImageFormat::Jpeg);
        }
        if lower.ends_with(".png") {
            return Some(ImageFormat::Png);
        }
    }
    image::guess_format(bytes).ok().and_then(|f| match f {
        ImageFormat::Jpeg | ImageFormat::Png => Some(f),
        _ => None,
    })
}

#[derive(Debug)]
pub struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    fn unsupported(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            message: msg.into(),
        }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
    fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}
