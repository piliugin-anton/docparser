//! Shared helpers for mmap safetensors loading and parity tensor dumps.

#![allow(unsafe_code)] // mmap safetensors in var_builder_from_safetensors

pub mod batch_norm;
mod error;
pub mod json_config;
pub mod lazy_runner;

use std::path::Path;

pub use json_config::{parse_id2label, read_json_file, read_json_from_dir};
pub use lazy_runner::{LazyRunner, LazyRunnerAccessError};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
pub use error::{CandleUtilsError, Result};

/// Matmul for MKL: both operands must be contiguous (transpose/views are not).
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> CandleResult<Tensor> {
    match (lhs.is_contiguous(), rhs.is_contiguous()) {
        (true, true) => lhs.matmul(rhs),
        (false, true) => lhs.contiguous()?.matmul(rhs),
        (true, false) => lhs.matmul(&rhs.contiguous()?),
        (false, false) => lhs.contiguous()?.matmul(&rhs.contiguous()?),
    }
}

/// `lhs @ transpose(rhs, dim1, dim2)`.
pub fn matmul_transpose(
    lhs: &Tensor,
    rhs: &Tensor,
    dim1: usize,
    dim2: usize,
) -> CandleResult<Tensor> {
    matmul(lhs, &rhs.transpose(dim1, dim2)?)
}

/// Back-compat alias for attention @ V.
pub fn matmul_contig_rhs(lhs: &Tensor, rhs: &Tensor) -> CandleResult<Tensor> {
    matmul(lhs, rhs)
}
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

/// Load a single `model.safetensors` via mmap into a [`VarBuilder`].
pub fn var_builder_from_safetensors(
    model_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let weights = model_dir.join("model.safetensors");
    if !weights.is_file() {
        return Err(CandleUtilsError::Message(format!(
            "missing weights at {}",
            weights.display()
        )));
    }
    // SAFETY: mmap is read-only; weights are not mutated.
    unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
        .map_err(CandleUtilsError::Candle)
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    let path = model_dir.join("model.safetensors");
    let bytes = std::fs::read(&path)
        .map_err(|e| CandleUtilsError::Message(format!("read {}: {e}", path.display())))?;
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
