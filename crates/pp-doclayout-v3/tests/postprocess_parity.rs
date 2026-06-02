use std::path::PathBuf;

use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use pp_doclayout_v3::LayoutModel;

fn layout_dir() -> PathBuf {
    workspace_root().join("models/PP-DocLayoutV3")
}

#[test]
#[ignore = "requires Candle layout port + fixture; set RUN_SLOW=1"]
fn layout_postprocess_matches_golden_count() {
    if !run_slow_enabled() {
        return;
    }
    let layout_dir = layout_dir();
    if !layout_dir.join("model.safetensors").is_file() {
        panic!("missing layout weights; run docparser-download");
    }
    let fixture = workspace_root().join("tests/fixtures/layout_demo.jpg");
    if !fixture.is_file() {
        panic!("missing fixture {}", fixture.display());
    }

    let model = LayoutModel::from_dir(&layout_dir).expect("load layout");
    let elements = model.detect_path(&fixture).expect("detect");
    let golden = load_golden_rel("tests/goldens/layout_postprocess.json");
    let expected_count = golden["detection_count"].as_u64().unwrap() as usize;
    assert_eq!(
        elements.len(),
        expected_count,
        "detection count mismatch"
    );
}
