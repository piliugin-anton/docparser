use std::path::PathBuf;

use docparser_test_utils::{assert_u32_ids_eq, load_golden_rel, run_slow_enabled, workspace_root};
use paddleocr_vl::{VlmModel, VlmTask};

const GOLDEN_REL: &str = "tests/goldens/vlm_generate_ocr_demo2.json";

fn fixture_path(name: &str) -> PathBuf {
    workspace_root().join("tests/fixtures").join(name)
}

fn require_vlm() -> (PathBuf, PathBuf) {
    let model_dir = workspace_root().join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing VLM weights; run docparser-download");
    }
    let golden = load_golden_rel(GOLDEN_REL);
    let fixture_name = golden["fixture"].as_str().expect("fixture");
    let image_path = fixture_path(fixture_name);
    if !image_path.is_file() {
        panic!("missing fixture {}", image_path.display());
    }
    (model_dir, image_path)
}

/// Single model load: greedy token ids + decoded text vs golden.
#[test]
#[ignore = "requires downloaded HF weights; set RUN_SLOW=1"]
fn generate_ocr_demo2_matches_golden() {
    if !run_slow_enabled() {
        return;
    }
    if !workspace_root().join(GOLDEN_REL).is_file() {
        panic!(
            "missing {GOLDEN_REL}; run: cargo run -p paddleocr-vl --bin vlm_write_golden --release"
        );
    }
    let golden = load_golden_rel(GOLDEN_REL);
    let (model_dir, image_path) = require_vlm();
    let max_new = golden["max_new_tokens"].as_u64().unwrap() as usize;
    let expected_text = golden["text"].as_str().expect("text");

    let device = docparser_candle_utils::device_from_env().unwrap_or(candle_core::Device::Cpu);
    let vlm = VlmModel::from_dir(&model_dir, device).expect("load vlm");
    let rgb = image::open(&image_path).unwrap().to_rgb8();
    let tokens = vlm
        .generate_token_ids(&rgb, VlmTask::Ocr, max_new)
        .expect("generate tokens");
    assert_u32_ids_eq(&tokens, &golden, "generated_token_ids");
    let text = vlm.decode_token_ids(&tokens).expect("decode");
    assert_eq!(text, expected_text, "decoded text mismatch");
}
