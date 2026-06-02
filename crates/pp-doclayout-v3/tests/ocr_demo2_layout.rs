use docparser_test_utils::{run_slow_enabled, workspace_root};
use pp_doclayout_v3::LayoutModel;

/// Layout on tiny `ocr_demo2.jpg` may score just below 0.5 with our resize; the pipeline uses a full-image fallback.
#[test]
#[ignore = "set RUN_SLOW=1"]
fn ocr_demo2_layout_may_be_empty_without_fallback() {
    if !run_slow_enabled() {
        return;
    }
    let layout_dir = workspace_root().join("models/PP-DocLayoutV3");
    let fixture = workspace_root().join("tests/fixtures/ocr_demo2.jpg");
    let model = LayoutModel::from_dir(&layout_dir).expect("load");
    let elements = model.detect_path(&fixture).expect("detect");
    eprintln!("ocr_demo2 layout-only detections: {}", elements.len());
}
