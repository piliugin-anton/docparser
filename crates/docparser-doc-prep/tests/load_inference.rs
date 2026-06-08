//! Slow inference smoke tests (requires downloaded weights under `models/` or env `DOC_PREP_MODELS_DIR`).

use std::path::PathBuf;

#[test]
#[ignore]
fn doc_orientation_classify_demo() {
    let base = model_base();
    let ori_dir = base.join("PP-LCNet_x1_0_doc_ori");
    if !ori_dir.join("model.safetensors").is_file() {
        eprintln!("skip: missing {}", ori_dir.display());
        return;
    }
    let model = docparser_doc_prep::orientation::DocOrientationModel::from_dir(
        &ori_dir,
        candle_core::Device::Cpu,
    )
    .expect("load ori");
    let img = image::open(fixture("ocr_demo2.jpg")).expect("open");
    let rgb = img.to_rgb8();
    let (angle, score) = model.classify(&rgb).expect("classify");
    assert!(score > 0.0);
    assert!(matches!(angle, 0 | 90 | 180 | 270));
    eprintln!("angle={angle} score={score:.4}");
}

#[test]
#[ignore]
fn uvdoc_rectify_demo() {
    let base = model_base();
    let uv_dir = base.join("UVDoc");
    if !uv_dir.join("model.safetensors").is_file() {
        eprintln!("skip: missing {}", uv_dir.display());
        return;
    }
    let model = docparser_doc_prep::unwarp::UvdocModel::from_dir(&uv_dir, candle_core::Device::Cpu)
        .expect("load uvdoc");
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
