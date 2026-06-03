use docparser_pipeline::{DocumentPipeline, PipelineConfig};
use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};

#[test]
#[ignore = "set RUN_SLOW=1"]
fn pipeline_smoke_layout_demo() {
    if !run_slow_enabled() {
        return;
    }
    let models_dir = workspace_root().join("models");
    if !models_dir
        .join("PaddleOCR-VL-1.6/model.safetensors")
        .is_file()
    {
        panic!("missing models; run docparser-download");
    }
    let fixture = workspace_root().join("tests/fixtures/layout_demo.jpg");
    let pipeline =
        DocumentPipeline::from_models_dir(&models_dir, PipelineConfig::default()).expect("pipeline");
    let result = pipeline.parse_path(&fixture).expect("parse");
    assert!(!result.blocks.is_empty());
    assert_eq!(result.pipeline_version, "v1.6");
    let md = result.markdown.unwrap_or_default();
    assert!(!md.is_empty() || result.blocks.iter().any(|b| !b.content.is_empty()));

    let golden = load_golden_rel("tests/goldens/pipeline/layout_demo_official.json");
    if golden["block_count"].as_u64().unwrap_or(0) > 0 {
        let expected = golden["block_count"].as_u64().unwrap() as usize;
        assert!(
            result.blocks.len() >= expected.saturating_sub(2),
            "block count {} vs expected ~{expected}",
            result.blocks.len()
        );
    }
}
