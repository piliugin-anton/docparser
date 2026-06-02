use docparser_pipeline::{DocumentPipeline, PipelineConfig, default_model_paths};
use docparser_test_utils::{load_golden_rel, run_slow_enabled, workspace_root};

#[test]
#[ignore = "requires full model stack; set RUN_SLOW=1"]
fn pipeline_snapshot_bounds() {
    if !run_slow_enabled() {
        return;
    }
    let models_dir = workspace_root().join("models");
    let (vl, layout) = default_model_paths(&models_dir);
    if !vl.join("model.safetensors").is_file() || !layout.join("model.safetensors").is_file() {
        panic!("missing models; run docparser-download");
    }
    let fixture = workspace_root().join("tests/fixtures/ocr_demo2.jpg");
    if !fixture.is_file() {
        panic!("missing fixture");
    }

    let pipeline = DocumentPipeline::from_dirs(
        vl,
        layout,
        PipelineConfig {
            max_tokens: 32,
            ..PipelineConfig::default()
        },
    )
    .expect("pipeline");

    let result = pipeline.parse_path(&fixture).expect("parse");
    let golden = load_golden_rel("tests/goldens/pipeline/page_simple.json");
    let min_blocks = golden["min_blocks"].as_u64().unwrap() as usize;
    let max_blocks = golden["max_blocks"].as_u64().unwrap() as usize;

    assert!(
        result.blocks.len() >= min_blocks && result.blocks.len() <= max_blocks,
        "block count {} outside [{min_blocks}, {max_blocks}]",
        result.blocks.len()
    );
    assert!(!result.blocks.is_empty());
    assert!(result.metadata.processing_ms > 0);
}
