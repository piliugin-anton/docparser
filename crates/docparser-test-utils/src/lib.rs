use serde_json::Value;

pub fn assert_close_f32(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(diff <= tol, "index {i}: {x} vs {y}, diff {diff} > {tol}");
    }
}

pub fn load_golden(path: &str) -> Value {
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read golden {path}: {e}"));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("parse golden {path}: {e}"))
}

pub fn run_slow_enabled() -> bool {
    std::env::var("RUN_SLOW")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Workspace root when called from a crate under `crates/<name>/`.
pub fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

pub fn load_golden_rel(rel: &str) -> Value {
    load_golden(
        workspace_root()
            .join(rel)
            .to_str()
            .expect("golden path utf8"),
    )
}

/// Element-wise comparison with absolute tolerance (scores, bboxes).
pub fn assert_slice_near(a: &[f32], b: &[f32], atol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= atol,
            "index {i}: {x} vs {y}, |diff|={diff} > atol {atol}"
        );
    }
}

pub fn assert_u32_ids_eq(actual: &[u32], golden: &Value, field: &str) {
    let expected: Vec<u32> = golden[field]
        .as_array()
        .unwrap_or_else(|| panic!("golden {field} array"))
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .collect();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{field} length {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "{field} mismatch at index {i}: {a} vs {e}");
    }
}

pub fn assert_input_ids_eq(actual: &[u32], golden: &Value) {
    let expected: Vec<u32> = golden["input_ids"]
        .as_array()
        .expect("golden input_ids array")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .collect();
    assert_eq!(
        actual.len(),
        expected.len(),
        "input_ids length {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "input_ids mismatch at index {i}: {a} vs {e}");
    }
}
