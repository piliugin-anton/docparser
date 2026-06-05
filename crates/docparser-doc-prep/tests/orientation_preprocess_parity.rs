use candle_core::{Device, IndexOp};
use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use docparser_doc_prep::orientation::{DocOrientationModel, PreprocessorConfig, preprocess};

#[test]
#[ignore = "set RUN_SLOW=1"]
fn doc_ori_preprocess_corners_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/PP-LCNet_x1_0_doc_ori");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing doc ori model; run docparser-download");
    }
    let golden = load_golden_rel("tests/goldens/doc_ori_preprocess.json");
    let prep_golden = golden["preprocess"].as_object().unwrap();
    let atol = golden["pixel_values_corner_atol"].as_f64().unwrap() as f32;
    let fixture = workspace_root()
        .join("tests/fixtures")
        .join(golden["fixture"].as_str().unwrap());
    let rgb = image::open(&fixture).unwrap().to_rgb8();
    let cfg = PreprocessorConfig::from_dir(&model_dir).expect("preprocessor cfg");
    let device = Device::Cpu;
    let tensor = preprocess(&rgb, &cfg, &device).expect("preprocess");
    let ch0 = tensor.i((0, 0)).expect("ch0").to_vec2::<f32>().unwrap();

    let corners = [
        ("top_left", 0, 0),
        ("top_right", 0, ch0[0].len() - 1),
        ("bottom_left", ch0.len() - 1, 0),
    ];
    for (name, y, x) in corners {
        let actual = ch0[y][x];
        let expected = prep_golden[name].as_f64().unwrap() as f32;
        assert!(
            (actual - expected).abs() <= atol,
            "{name}: {actual} vs {expected}"
        );
    }
}

#[test]
#[ignore = "set RUN_SLOW=1"]
fn doc_ori_classify_and_logits_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/PP-LCNet_x1_0_doc_ori");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing doc ori model; run docparser-download");
    }
    let golden = load_golden_rel("tests/goldens/doc_ori_preprocess.json");
    let fixture = workspace_root()
        .join("tests/fixtures")
        .join(golden["fixture"].as_str().unwrap());
    let expected_angle = golden["predicted_angle"].as_u64().unwrap() as u32;
    let expected_logits: Vec<f32> = golden["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let atol = golden["logits_atol"].as_f64().unwrap() as f32;

    let model = DocOrientationModel::from_dir(&model_dir).expect("load");
    let rgb = image::open(&fixture).expect("open").to_rgb8();
    let (angle, score) = model.classify(&rgb).expect("classify");
    assert_eq!(angle, expected_angle);
    assert!(score > 0.0);

    let logits = model.logits(&rgb).expect("logits");
    assert_eq!(logits.len(), expected_logits.len());
    for (i, (&actual, &expected)) in logits.iter().zip(expected_logits.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= atol,
            "logit[{i}]: {actual} vs {expected}"
        );
    }
}
