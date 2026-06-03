//! Slow inference smoke test (requires downloaded weights under `models/` or env `DOC_PREP_MODELS_DIR`).

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
    let model = pp_lcnet_doc_ori::DocOrientationModel::from_dir(&ori_dir).expect("load ori");
    let img = image::open(fixture("ocr_demo2.jpg")).expect("open");
    let rgb = img.to_rgb8();
    let (angle, score) = model.classify(&rgb).expect("classify");
    assert!(score > 0.0);
    assert!(matches!(angle, 0 | 90 | 180 | 270));
    eprintln!("angle={angle} score={score:.4}");
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
