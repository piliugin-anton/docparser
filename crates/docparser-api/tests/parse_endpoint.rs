use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use docparser_api::{ApiConfig, AppState, build_router};
use docparser_test_utils::{run_slow_enabled, workspace_root};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn models_available() -> bool {
    workspace_root()
        .join("models/PaddleOCR-VL-1.6/model.safetensors")
        .is_file()
        && workspace_root()
            .join("models/PP-DocLayoutV3/model.safetensors")
            .is_file()
}

#[test]
fn health_route_returns_ok_when_models_loaded() {
    if !models_available() {
        eprintln!("skip health integration test: models not downloaded");
        return;
    }

    let mut config = ApiConfig::default();
    config.models_dir = workspace_root().join("models");
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
    if !models_available() {
        eprintln!("skip parse test: models not downloaded");
        return;
    }

    let mut config = ApiConfig::default();
    config.models_dir = workspace_root().join("models");
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

#[test]
#[ignore = "full inference; set RUN_SLOW=1"]
fn parse_ocr_demo2_returns_blocks() {
    if !run_slow_enabled() || !models_available() {
        return;
    }
    let fixture = workspace_root().join("tests/fixtures/ocr_demo2.jpg");
    if !fixture.is_file() {
        panic!("missing fixture {}", fixture.display());
    }

    let mut config = ApiConfig::default();
    config.models_dir = workspace_root().join("models");
    let (vl, layout) = docparser_pipeline::default_model_paths(&config.models_dir);
    let pipeline = docparser_pipeline::DocumentPipeline::from_dirs(
        vl,
        layout,
        docparser_pipeline::PipelineConfig {
            max_tokens: 32,
            ..Default::default()
        },
    )
    .expect("load pipeline");

    let state = AppState {
        pipeline: Arc::new(Mutex::new(pipeline)),
    };
    let app = build_router(state, config.max_upload_mb * 1024 * 1024);

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let boundary = "testboundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ocr_demo2.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n",
    );
    let mut body_bytes = body.into_bytes();
    body_bytes.extend_from_slice(&bytes);
    body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/parse")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body_bytes))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(json.get("error").is_none(), "unexpected error: {json}");
        let blocks = json["blocks"].as_array().expect("blocks array");
        assert!(!blocks.is_empty(), "expected at least one block");
    });
}
