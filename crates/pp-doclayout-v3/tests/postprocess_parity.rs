use std::path::PathBuf;

use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use pp_doclayout_v3::{label_name, LayoutModel};

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

    let expected_labels: Vec<i64> = golden["labels"]
        .as_array()
        .expect("labels array")
        .iter()
        .map(|v| v.as_i64().expect("label id"))
        .collect();
    assert_eq!(expected_labels.len(), elements.len());
    for (el, &lid) in elements.iter().zip(expected_labels.iter()) {
        assert_eq!(
            el.label,
            label_name(lid),
            "label mismatch for id {lid}"
        );
    }
}
