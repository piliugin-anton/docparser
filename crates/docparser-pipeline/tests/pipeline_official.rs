use docparser_pipeline::{DocumentPipeline, PipelineConfig};
use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};

#[test]
#[ignore = "set RUN_SLOW=1"]
fn pipeline_official_v16_smoke_layout_demo() {
    if !run_slow_enabled() {
        return;
    }
    let models_dir = workspace_root().join("models");
    let (vl, layout) = docparser_pipeline::default_model_paths(&models_dir);
    if !vl.join("model.safetensors").is_file() {
        panic!("missing models");
    }
    let fixture = workspace_root().join("tests/fixtures/layout_demo.jpg");
    let pipeline =
        DocumentPipeline::from_dirs(vl, layout, PipelineConfig::official_v16()).expect("pipeline");
    let result = pipeline.parse_path(&fixture).expect("parse");
    assert!(!result.blocks.is_empty());
    assert_eq!(result.pipeline_version, "v1.6-official");
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
