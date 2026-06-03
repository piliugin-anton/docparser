use docparser_test_utils::{assert_input_ids_eq, load_golden_rel, run_slow_enabled, workspace_root};
use paddleocr_vl::VlmTask;

#[test]
fn task_prompt_mapping() {
    assert_eq!(paddleocr_vl::task_for_layout_label("table").prompt(), "Table Recognition:");
    assert_eq!(paddleocr_vl::task_for_layout_label("text").prompt(), "OCR:");
    assert_eq!(
        paddleocr_vl::task_for_layout_label("display_formula").prompt(),
        "Formula Recognition:"
    );
}

#[test]
fn should_run_vlm_gating() {
    assert!(paddleocr_vl::should_run_vlm_for_label(
        "footnote",
        false,
        false,
        false
    ));
    assert!(!paddleocr_vl::should_run_vlm_for_label(
        "chart",
        false,
        false,
        false
    ));
    assert!(paddleocr_vl::should_run_vlm_for_label(
        "text",
        false,
        false,
        false
    ));
    assert!(!paddleocr_vl::should_run_vlm_for_label(
        "image",
        false,
        false,
        false
    ));
    assert!(paddleocr_vl::should_run_vlm_for_label(
        "image",
        false,
        false,
        true
    ));
}

#[test]
#[ignore = "requires downloaded VLM weights; set RUN_SLOW=1"]
fn preprocess_golden_values() {
    if !run_slow_enabled() {
        return;
    }
    let golden = load_golden_rel("tests/goldens/vlm_preprocess_ocr_demo2.json");
    assert_eq!(golden["prompt"].as_str(), Some("OCR:"));

    let model_dir = workspace_root().join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing VLM weights");
    }
    let image_path = workspace_root().join("tests/fixtures/ocr_demo2.jpg");
    let device = candle_core::Device::Cpu;
    let vlm = paddleocr_vl::VlmModel::from_dir(&model_dir, device).expect("load");
    let rgb = image::open(&image_path).unwrap().to_rgb8();
    let ids = vlm
        .preprocess_input_ids(&rgb, VlmTask::Ocr)
        .expect("preprocess");
    assert_eq!(ids.len(), golden["input_ids_len"].as_u64().unwrap() as usize);
    if golden.get("input_ids").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty()) {
        assert_input_ids_eq(&ids, &golden);
    }

    if let Some(expected_grid) = golden.get("grid_thw").and_then(|v| v.as_array()) {
        let grid = vlm.preprocess_grid_thw(&rgb, VlmTask::Ocr).expect("grid_thw");
        assert_eq!(grid.len(), expected_grid.len(), "grid_thw batch size");
        for (row, exp_row) in grid.iter().zip(expected_grid.iter()) {
            let exp: Vec<u32> = exp_row
                .as_array()
                .expect("grid row")
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(row, &exp, "grid_thw row");
        }
    }
}

#[test]
#[ignore = "set RUN_SLOW=1"]
fn preprocess_tasks_match_golden_len() {
    if !run_slow_enabled() {
        return;
    }
    let tasks = load_golden_rel("tests/goldens/vlm_preprocess_tasks.json");
    let model_dir = workspace_root().join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        return;
    }
    let rgb = image::open(workspace_root().join("tests/fixtures/ocr_demo2.jpg"))
        .unwrap()
        .to_rgb8();
    let vlm = paddleocr_vl::VlmModel::from_dir(&model_dir, candle_core::Device::Cpu).unwrap();
    for (key, task) in [
        ("ocr", VlmTask::Ocr),
        ("table", VlmTask::Table),
        ("formula", VlmTask::Formula),
    ] {
        let g = &tasks[key];
        let expected_len = g["input_ids_len"].as_u64().unwrap_or(0) as usize;
        if expected_len == 0 {
            continue;
        }
        let len = vlm.preprocess_input_ids(&rgb, task).unwrap().len();
        assert_eq!(len, expected_len, "task {key}");
    }
}
