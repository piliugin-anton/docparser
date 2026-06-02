use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use paddleocr_vl::VlmTask;

#[test]
fn task_prompt_mapping() {
    assert_eq!(paddleocr_vl::task_for_layout_label("table").prompt(), "Table Recognition:");
    assert_eq!(paddleocr_vl::task_for_layout_label("text").prompt(), "OCR:");
}

#[test]
#[ignore = "requires downloaded VLM weights; set RUN_SLOW=1"]
fn preprocess_golden_values() {
    if !run_slow_enabled() {
        return;
    }
    let golden = load_golden_rel("tests/goldens/vlm_preprocess_ocr_demo2.json");
    assert_eq!(golden["prompt"].as_str(), Some("OCR:"));
    let _ = VlmTask::Ocr.prompt();

    let model_dir = workspace_root().join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing VLM weights");
    }
    let image_path = workspace_root().join("tests/fixtures/ocr_demo2.jpg");
    let device = candle_core::Device::Cpu;
    let vlm = paddleocr_vl::VlmModel::from_dir(&model_dir, device).expect("load");
    let rgb = image::open(&image_path).unwrap().to_rgb8();
    let len = vlm
        .preprocess_input_ids_len(&rgb, VlmTask::Ocr)
        .expect("preprocess");
    assert_eq!(len, golden["input_ids_len"].as_u64().unwrap() as usize);
}
