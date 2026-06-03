//! Slow inference smoke test (requires downloaded weights).

use std::path::PathBuf;

#[test]
#[ignore]
fn uvdoc_rectify_demo() {
    let base = model_base();
    let uv_dir = base.join("UVDoc");
    if !uv_dir.join("model.safetensors").is_file() {
        eprintln!("skip: missing {}", uv_dir.display());
        return;
    }
    let model = uvdoc::UvdocModel::from_dir(&uv_dir).expect("load uvdoc");
    let img = image::open(fixture("ocr_demo2.jpg"))
        .expect("open")
        .to_rgb8();
    let out = model.rectify(&img).expect("rectify");
    assert!(out.width() > 0 && out.height() > 0);
    eprintln!("rectified size={:?}", out.dimensions());
}

fn model_base() -> PathBuf {
    std::env::var("DOC_PREP_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/docparser-models-test"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}
