use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use paddleocr_vl::VlmTask;

#[test]
fn task_prompt_mapping() {
    assert_eq!(paddleocr_vl::task_for_layout_label("table").prompt(), "Table Recognition:");
    assert_eq!(paddleocr_vl::task_for_layout_label("text").prompt(), "OCR:");
}

#[test]
#[ignore = "requires processor parity harness; set RUN_SLOW=1"]
fn preprocess_golden_values() {
    if !run_slow_enabled() {
        return;
    }
    let golden = load_golden_rel("tests/goldens/vlm_preprocess_ocr_demo2.json");
    assert_eq!(golden["input_ids_len"].as_u64(), Some(211));
    assert_eq!(golden["prompt"].as_str(), Some("OCR:"));
    let _ = VlmTask::Ocr.prompt();
}
