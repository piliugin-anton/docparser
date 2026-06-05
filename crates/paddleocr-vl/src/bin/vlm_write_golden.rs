//! Write `tests/goldens/vlm_generate_ocr_demo2.json` from Rust greedy decode (dev-only).
//!
//! Prefer regenerating via HF when available:
//!   python tools/parity_gen.py --update-goldens --vlm

use std::path::PathBuf;
use std::process;

use paddleocr_vl::{Result, VlmError, VlmModel, VlmTask};
use serde_json::json;
use sha2::{Digest, Sha256};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run() -> Result<()> {
    let root = workspace_root();
    let golden_path = root.join("tests/goldens/vlm_generate_ocr_demo2.json");
    let fixture_name = "ocr_demo2.jpg";
    let image_path = root.join("tests/fixtures").join(fixture_name);
    let model_dir = root.join("models/PaddleOCR-VL-1.6");
    if !model_dir.join("model.safetensors").is_file() {
        return Err(VlmError::Message(format!(
            "missing {}; run docparser-download",
            model_dir.display()
        )));
    }

    let max_new_tokens = 30usize;
    let device = candle_core::Device::Cpu;
    let vlm = VlmModel::from_dir(&model_dir, device)?;
    let rgb = image::open(&image_path).map_err(VlmError::Image)?.to_rgb8();
    let input_ids_len = vlm.preprocess_input_ids(&rgb, VlmTask::Ocr)?.len();
    let tokens = vlm.generate_token_ids(&rgb, VlmTask::Ocr, max_new_tokens)?;
    let text = vlm.decode_token_ids(&tokens)?;
    let preprocess = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(
        root.join("tests/goldens/vlm_preprocess_ocr_demo2.json"),
    )?)?;
    let eos_token_id = preprocess["generation_config"]["eos_token_id"]
        .as_u64()
        .unwrap_or(2) as u32;

    let text_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));

    let payload = json!({
        "fixture": fixture_name,
        "task": "ocr",
        "prompt": "OCR:",
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "input_ids_len": input_ids_len,
        "generated_token_ids": tokens,
        "text": text,
        "text_sha256": text_sha256,
        "source": "rust_greedy",
    });

    std::fs::write(&golden_path, serde_json::to_string_pretty(&payload)? + "\n")?;
    println!("wrote {}", golden_path.display());
    println!(
        "tokens={} text_len={}",
        payload["generated_token_ids"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        text.len()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
