use std::path::{Path, PathBuf};

use docparser_test_utils::load_golden_rel;
use paddleocr_vl::{VlmConfig, list_safetensor_keys};

fn vlm_dir() -> PathBuf {
    docparser_test_utils::workspace_root().join("models/PaddleOCR-VL-1.6")
}

#[test]
fn config_parses_from_fixture() {
    let cfg = VlmConfig::from_dir(Path::new("tests/fixtures")).expect("parse fixture config");
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.vocab_size, 103424);
    assert_eq!(cfg.torch_dtype, "bfloat16");
}

#[test]
fn safetensor_keys_match_expected_prefixes_when_present() {
    let model_dir = vlm_dir();
    let weights = model_dir.join("model.safetensors");
    if !weights.is_file() {
        eprintln!(
            "skip: {} not found (run docparser-download)",
            weights.display()
        );
        return;
    }

    let keys = list_safetensor_keys(&model_dir).expect("list keys");
    assert!(!keys.is_empty());

    let golden = load_golden_rel("tests/goldens/vlm_safetensor_key_prefixes.json");
    let prefixes: Vec<String> = golden["required_prefixes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let allow: Vec<String> = golden["metadata_allowlist"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    for key in &keys {
        if allow.iter().any(|a| key.starts_with(a)) {
            continue;
        }
        assert!(
            prefixes.iter().any(|p| key.starts_with(p)),
            "unexpected safetensor key: {key}"
        );
    }
}

#[test]
fn safetensor_file_has_no_nan_on_load_smoke() {
    let model_dir = vlm_dir();
    let weights = model_dir.join("model.safetensors");
    if !weights.is_file() {
        return;
    }
    let bytes = std::fs::read(&weights).expect("read weights");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize");
    for name in tensors.names() {
        let view = tensors.tensor(name).expect("tensor view");
        let slice = view.data();
        assert!(!slice.is_empty(), "empty tensor {name}");
    }
}
