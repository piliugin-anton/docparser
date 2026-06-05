use candle_core::{Device, IndexOp};
use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use uvdoc::{PreprocessorConfig, UvdocModel, preprocess};

#[test]
#[ignore = "set RUN_SLOW=1"]
fn uvdoc_preprocess_corners_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/UVDoc");
    if !model_dir.join("model.safetensors").is_file() {
        panic!("missing UVDoc model; run docparser-download");
    }
    let golden = load_golden_rel("tests/goldens/uvdoc_preprocess.json");
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
fn uvdoc_flow_corners_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/UVDoc");
    let golden = load_golden_rel("tests/goldens/uvdoc_preprocess.json");
    let prep_golden = golden["flow_corners"].as_object().unwrap();
    let atol = golden["flow_atol"].as_f64().unwrap() as f32;
    let fixture = workspace_root()
        .join("tests/fixtures")
        .join(golden["fixture"].as_str().unwrap());
    let rgb = image::open(&fixture).unwrap().to_rgb8();

    let model = UvdocModel::from_dir(&model_dir).expect("load");
    let flow = model.forward_flow(&rgb).expect("flow");
    let flow_ch0 = flow.i((0, 0)).expect("ch0").to_vec2::<f32>().unwrap();
    let fh = flow_ch0.len();
    let fw = flow_ch0[0].len();
    let samples = [
        ("top_left", 0, 0),
        ("top_right", 0, fw - 1),
        ("bottom_left", fh - 1, 0),
    ];
    for (name, y, x) in samples {
        let actual = flow_ch0[y][x];
        let expected = prep_golden[name].as_f64().unwrap() as f32;
        assert!(
            (actual - expected).abs() <= atol,
            "flow {name}: {actual} vs {expected}"
        );
    }
}

#[test]
#[ignore = "set RUN_SLOW=1"]
fn uvdoc_rectify_corners_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/UVDoc");
    let golden_path = workspace_root().join("tests/goldens/uvdoc_rectify.json");
    if !model_dir.join("model.safetensors").is_file() || !golden_path.is_file() {
        panic!("missing UVDoc model or golden; run docparser-download and parity_gen --doc-prep");
    }
    let golden = load_golden_rel("tests/goldens/uvdoc_rectify.json");
    let prep_golden = load_golden_rel("tests/goldens/uvdoc_preprocess.json");
    let fixture = workspace_root()
        .join("tests/fixtures")
        .join(prep_golden["fixture"].as_str().unwrap());
    let corners = golden["corners_rgb"].as_object().unwrap();
    let atol = golden["pixel_atol"].as_u64().unwrap() as i32;
    let expected_size: Vec<u32> = golden["output_size"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let model = UvdocModel::from_dir(&model_dir).expect("load");
    let rgb = image::open(&fixture).expect("open").to_rgb8();
    let out = model.rectify(&rgb).expect("rectify");
    assert_eq!(out.dimensions(), (expected_size[0], expected_size[1]));

    let samples = [
        ("top_left", 0, 0),
        ("top_right", out.width() - 1, 0),
        ("bottom_left", 0, out.height() - 1),
    ];
    for (name, x, y) in samples {
        let p = out.get_pixel(x, y);
        let expected: Vec<i32> = corners[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        for c in 0..3 {
            assert!(
                (p[c] as i32 - expected[c]).abs() <= atol,
                "{name} ch{c}: {} vs {}",
                p[c],
                expected[c]
            );
        }
    }
}
