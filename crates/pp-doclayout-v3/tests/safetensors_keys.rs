use std::path::Path;

use pp_doclayout_v3::{LayoutConfig, list_safetensor_keys};

#[test]
fn config_parses_from_fixture() {
    let cfg = LayoutConfig::from_dir(Path::new("tests/fixtures"), 0.5).expect("parse layout config");
    assert_eq!(cfg.num_queries, 300);
    assert_eq!(cfg.num_labels, 25);
}

#[test]
fn safetensor_keys_match_expected_prefixes_when_present() {
    let model_dir = docparser_test_utils::workspace_root().join("models/PP-DocLayoutV3");
    let weights = model_dir.join("model.safetensors");
    if !weights.is_file() {
        eprintln!("skip: {} not found", weights.display());
        return;
    }

    let keys = list_safetensor_keys(&model_dir).expect("list keys");
    assert!(!keys.is_empty());

    let golden =
        docparser_test_utils::load_golden_rel("tests/goldens/layout_safetensor_key_prefixes.json");
    let prefixes: Vec<String> = golden["required_prefixes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    for key in &keys {
        assert!(
            prefixes.iter().any(|p| key.starts_with(p)),
            "unexpected safetensor key: {key}"
        );
    }
}
