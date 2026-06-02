//! Shared helpers for mmap safetensors loading and parity tensor dumps.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

/// Load a single `model.safetensors` via mmap into a [`VarBuilder`].
pub fn var_builder_from_safetensors(
    model_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let weights = model_dir.join("model.safetensors");
    anyhow::ensure!(
        weights.is_file(),
        "missing weights at {}",
        weights.display()
    );
    let weights_path = weights.clone();
    // SAFETY: mmap is read-only; weights are not mutated.
    unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device) }
        .with_context(|| format!("mmap safetensors {}", weights.display()))
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    let path = model_dir.join("model.safetensors");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let data = SafeTensors::deserialize(&bytes)?;
    let mut keys: Vec<String> = data.names().into_iter().map(str::to_string).collect();
    keys.sort();
    Ok(keys)
}

/// Optional intermediate tensor dump for Python parity scripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDump {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub fn tensor_dump_f32(t: &candle_core::Tensor, name: impl Into<String>) -> Result<TensorDump> {
    let shape = t.dims().to_vec();
    let values = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(TensorDump {
        name: name.into(),
        shape,
        values,
    })
}

pub fn write_tensor_dumps(path: &Path, dumps: &[TensorDump]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(dumps)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn default_onnx_layout_path(model_dir: &Path) -> PathBuf {
    for candidate in [
        model_dir.join("inference.onnx"),
        model_dir.join("model.onnx"),
        model_dir
            .parent()
            .map(|p| p.join("PP-DocLayoutV3-onnx").join("PP-DocLayoutV3.onnx"))
            .unwrap_or_default(),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    model_dir.join("inference.onnx")
}
