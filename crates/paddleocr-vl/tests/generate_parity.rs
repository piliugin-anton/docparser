use std::path::{Path, PathBuf};

use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use paddleocr_vl::{VlmModel, VlmTask};

fn fixture_path(name: &str) -> PathBuf {
    workspace_root().join("tests/fixtures").join(name)
}

#[test]
#[ignore = "requires downloaded HF weights; set RUN_SLOW=1"]
fn preprocess_golden_schema_present() {
    if !run_slow_enabled() {
        return;
    }
    let golden = load_golden_rel("tests/goldens/vlm_preprocess_ocr_demo2.json");
    assert_eq!(golden["input_ids_len"].as_u64().unwrap(), 211);
    assert_eq!(golden["prompt"].as_str().unwrap(), "OCR:");
}

#[test]
#[ignore = "requires downloaded HF weights; set RUN_SLOW=1"]
fn generate_ocr_demo2_contains_golden_text() {
    if !run_slow_enabled() {
        return;
    }
    let image_path = fixture_path("ocr_demo2.jpg");
    if !image_path.is_file() {
        panic!("missing fixture {}", image_path.display());
    }
    let model_dir = workspace_root().join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing VLM weights; run docparser-download");
    }

    let device = candle_core::Device::Cpu;
    let vlm = VlmModel::from_dir(&model_dir, device).expect("load vlm");
    let text = vlm
        .generate_from_path(&image_path, VlmTask::Ocr, 30)
        .expect("generate");
    assert!(
        text.contains("生甘草"),
        "expected 生甘草 in output, got: {text}"
    );
}
