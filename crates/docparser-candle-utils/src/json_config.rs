//! Helpers for reading HuggingFace-style JSON config files from model directories.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::{CandleUtilsError, Result};

/// Read a UTF-8 JSON file from disk.
pub fn read_json_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(CandleUtilsError::from)
}

/// Read `filename` from `model_dir` (e.g. `config.json`, `preprocessor_config.json`).
pub fn read_json_from_dir(model_dir: &Path, filename: &str) -> Result<String> {
    let path = model_dir.join(filename);
    read_json_file(&path).map_err(|e| match e {
        CandleUtilsError::Io(err) => CandleUtilsError::Message(format!(
            "read {}: {err}",
            path.display()
        )),
        other => other,
    })
}

/// Parse HF `id2label` object keys into a `u32 -> label` map.
pub fn parse_id2label(map: &serde_json::Map<String, Value>) -> HashMap<u32, String> {
    map.iter()
        .filter_map(|(k, v)| {
            let id: u32 = k.parse().ok()?;
            let name = v.as_str()?.to_string();
            Some((id, name))
        })
        .collect()
}
