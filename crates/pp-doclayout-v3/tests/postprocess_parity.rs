use std::path::PathBuf;

use docparser_test_utils::{assert_slice_near, load_golden_rel, run_slow_enabled, workspace_root};
use pp_doclayout_v3::LayoutModel;

fn layout_dir() -> PathBuf {
    workspace_root().join("models/PP-DocLayoutV3")
}

fn run_layout_parity(fixture: &str, golden_rel: &str) {
    if !run_slow_enabled() {
        return;
    }
    let layout_dir = layout_dir();
    if !layout_dir.join("model.safetensors").is_file() {
        panic!("missing layout weights; run docparser-download");
    }
    let fixture_path = workspace_root().join("tests/fixtures").join(fixture);
    if !fixture_path.is_file() {
        panic!("missing fixture {}", fixture_path.display());
    }

    let model = LayoutModel::from_dir(&layout_dir).expect("load layout");
    let elements = model.detect_path(&fixture_path).expect("detect");
    let golden = load_golden_rel(golden_rel);
    let expected_count = golden["detection_count"].as_u64().unwrap() as usize;
    assert_eq!(
        elements.len(),
        expected_count,
        "detection count mismatch for {fixture}"
    );

    let score_atol = golden["score_atol"].as_f64().unwrap() as f32;
    let bbox_atol = golden["bbox_atol"].as_f64().unwrap() as f32;

    if let Some(dets) = golden["detections"].as_array() {
        if !dets.is_empty() {
            assert_eq!(dets.len(), elements.len());
            for (el, det) in elements.iter().zip(dets.iter()) {
                let lid = det["label"].as_i64().unwrap();
                assert_eq!(
                    el.label,
                    model.config().label_for_id(lid),
                    "label mismatch for id {lid}"
                );
                assert_slice_near(&[el.score], &[det["score"].as_f64().unwrap() as f32], score_atol);
                let bbox: Vec<f32> = det["bbox"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                assert_slice_near(&el.bbox, &bbox, bbox_atol);
            }
            return;
        }
    }

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
            model.config().label_for_id(lid),
            "label mismatch for id {lid}"
        );
    }
    if let Some(min_score) = golden["first_score_min"].as_f64() {
        assert!(elements[0].score >= min_score as f32);
    }
}

#[test]
#[ignore = "requires Candle layout port + fixture; set RUN_SLOW=1"]
fn layout_postprocess_matches_golden_count() {
    run_layout_parity("layout_demo.jpg", "tests/goldens/layout_postprocess.json");
}

#[test]
#[ignore = "set RUN_SLOW=1"]
fn layout_postprocess_ocr_demo2() {
    run_layout_parity("ocr_demo2.jpg", "tests/goldens/layout_postprocess_ocr_demo2.json");
}
