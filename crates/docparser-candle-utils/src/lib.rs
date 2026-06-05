//! Shared helpers for mmap safetensors loading and parity tensor dumps.

#![allow(unsafe_code)] // mmap safetensors in var_builder_from_safetensors

mod error;

use std::path::Path;

pub use error::{CandleUtilsError, Result};
use candle_core::{Device, DType, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;

fn contig(t: &Tensor) -> CandleResult<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Matmul for MKL: both operands must be contiguous (transpose/views are not).
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> CandleResult<Tensor> {
    contig(lhs)?.matmul(&contig(rhs)?)
}

/// `lhs @ transpose(rhs, dim1, dim2)`.
pub fn matmul_transpose(lhs: &Tensor, rhs: &Tensor, dim1: usize, dim2: usize) -> CandleResult<Tensor> {
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
    let weights_path = weights.clone();
    // SAFETY: mmap is read-only; weights are not mutated.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device) }
        .map_err(CandleUtilsError::Candle)?)
}

pub fn list_safetensor_keys(model_dir: &Path) -> Result<Vec<String>> {
    let path = model_dir.join("model.safetensors");
    let bytes = std::fs::read(&path).map_err(|e| {
        CandleUtilsError::Message(format!("read {}: {e}", path.display()))
    })?;
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
