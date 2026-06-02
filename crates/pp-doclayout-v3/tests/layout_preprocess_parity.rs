use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};
use pp_doclayout_v3::LayoutImageProcessor;

#[test]
#[ignore = "set RUN_SLOW=1"]
fn layout_preprocess_corners_match_golden() {
    if !run_slow_enabled() {
        return;
    }
    let model_dir = workspace_root().join("models/PP-DocLayoutV3");
    if !model_dir.join("preprocessor_config.json").is_file() {
        panic!("missing layout model; run docparser-download");
    }
    let golden = load_golden_rel("tests/goldens/layout_postprocess.json");
    let prep_golden = golden["preprocess"]["pixel_values"].as_object();
    if prep_golden.is_none() || prep_golden.unwrap().is_empty() {
        return;
    }
    let prep_golden = golden["preprocess"]["pixel_values"].as_object().unwrap();
    let atol = 1e-3_f32;

    let fixture = workspace_root().join("tests/fixtures/layout_demo.jpg");
    let rgb = image::open(&fixture).unwrap().to_rgb8();
    let device = candle_core::Device::Cpu;
    let processor = LayoutImageProcessor::from_dir(&model_dir).expect("processor");
    let out = processor.preprocess(&rgb, &device).expect("preprocess");
    let pv = out.pixel_values.to_vec3::<f32>().unwrap();

    let corners = [
        ("top_left", 0, 0),
        ("top_right", 0, pv[0][0].len() - 1),
        ("bottom_left", pv[0].len() - 1, 0),
    ];
    for (name, y, x) in corners {
        let actual = pv[0][y][x];
        let expected = prep_golden[name].as_f64().unwrap() as f32;
        assert!(
            (actual - expected).abs() <= atol,
            "{name}: {actual} vs {expected}"
        );
    }
}
