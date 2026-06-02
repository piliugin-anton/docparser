use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use docparser_api::{ApiConfig, AppState, build_router};
use docparser_test_utils::workspace_root;
use http_body_util::BodyExt;
use tower::ServiceExt;

#[test]
fn health_route_returns_ok_when_models_loaded() {
    if !workspace_root()
        .join("models/PaddleOCR-VL-1.6/model.safetensors")
        .is_file()
    {
        eprintln!("skip health integration test: models not downloaded");
        return;
    }

    let config = ApiConfig::default();
    let (vl, layout) = docparser_pipeline::default_model_paths(&config.models_dir);
    let pipeline = docparser_pipeline::DocumentPipeline::from_dirs(
        vl,
        layout,
        docparser_pipeline::PipelineConfig::default(),
    )
    .expect("load pipeline");

    let state = AppState {
        pipeline: Arc::new(Mutex::new(pipeline)),
    };
    let app = build_router(state, 20 * 1024 * 1024);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    });
}

#[test]
fn parse_rejects_missing_file_field() {
    if !workspace_root()
        .join("models/PaddleOCR-VL-1.6/model.safetensors")
        .is_file()
    {
        eprintln!("skip parse test: models not downloaded");
        return;
    }

    let config = ApiConfig::default();
    let (vl, layout) = docparser_pipeline::default_model_paths(&config.models_dir);
    let pipeline = docparser_pipeline::DocumentPipeline::from_dirs(
        vl,
        layout,
        docparser_pipeline::PipelineConfig::default(),
    )
    .expect("load pipeline");

    let state = AppState {
        pipeline: Arc::new(Mutex::new(pipeline)),
    };
    let app = build_router(state, config.max_upload_mb * 1024 * 1024);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let boundary = "boundary123";
        let body = format!("--{boundary}--\r\n");
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/parse")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    });
}
