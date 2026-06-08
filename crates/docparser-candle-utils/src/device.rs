//! Candle device selection from the `BACKEND` environment variable.

use candle_core::Device;

use crate::{CandleUtilsError, Result};

/// Inference backend requested via `BACKEND`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Auto,
}

impl BackendKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "auto" => Ok(Self::Auto),
            other => Err(CandleUtilsError::Message(format!(
                "invalid BACKEND={other:?}; expected cpu, cuda, metal, or auto"
            ))),
        }
    }
}

fn default_backend_kind() -> BackendKind {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        BackendKind::Auto
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        BackendKind::Cpu
    }
}

/// Read `BACKEND` from the environment.
///
/// When unset: `auto` if a GPU cargo feature is enabled, otherwise `cpu`.
pub fn backend_from_env() -> Result<BackendKind> {
    match std::env::var("BACKEND") {
        Ok(value) => BackendKind::parse(&value),
        Err(std::env::VarError::NotPresent) => Ok(default_backend_kind()),
        Err(e) => Err(CandleUtilsError::Message(format!(
            "failed to read BACKEND: {e}"
        ))),
    }
}

/// Resolve the Candle [`Device`] from `BACKEND` (see [`backend_from_env`]).
pub fn device_from_env() -> Result<Device> {
    resolve_device(backend_from_env()?)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn device_ordinal_from_env(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn backend_not_compiled(name: &str, feature: &str) -> CandleUtilsError {
    CandleUtilsError::Message(format!(
        "BACKEND={name}, but the `{feature}` cargo feature was not enabled; rebuild with --features {feature}"
    ))
}

/// Resolve a [`Device`] for the given backend kind.
pub fn resolve_device(kind: BackendKind) -> Result<Device> {
    match kind {
        BackendKind::Cpu => Ok(Device::Cpu),
        BackendKind::Cuda => resolve_cuda(),
        BackendKind::Metal => resolve_metal(),
        BackendKind::Auto => resolve_auto_device(),
    }
}

fn resolve_cuda() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let ordinal = device_ordinal_from_env("CUDA_DEVICE");
        Device::new_cuda(ordinal).map_err(|e| {
            CandleUtilsError::Message(format!(
                "BACKEND=cuda, but no CUDA device is available (CUDA_DEVICE={ordinal}): {e}"
            ))
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(backend_not_compiled("cuda", "cuda"))
    }
}

fn resolve_metal() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        let ordinal = device_ordinal_from_env("METAL_DEVICE");
        Device::new_metal(ordinal).map_err(|e| {
            CandleUtilsError::Message(format!(
                "BACKEND=metal, but no Metal device is available (METAL_DEVICE={ordinal}): {e}"
            ))
        })
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(backend_not_compiled("metal", "metal"))
    }
}

fn resolve_auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let cuda_ordinal = device_ordinal_from_env("CUDA_DEVICE");
        if candle_core::utils::cuda_is_available() {
            if let Ok(device) = Device::new_cuda(cuda_ordinal) {
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        let metal_ordinal = device_ordinal_from_env("METAL_DEVICE");
        if candle_core::utils::metal_is_available() {
            if let Ok(device) = Device::new_metal(metal_ordinal) {
                return Ok(device);
            }
        }
    }

    Ok(Device::Cpu)
}

/// Short label for API metadata and logging.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_values() {
        assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("CUDA").unwrap(), BackendKind::Cuda);
        assert_eq!(BackendKind::parse("metal").unwrap(), BackendKind::Metal);
        assert_eq!(BackendKind::parse("auto").unwrap(), BackendKind::Auto);
        assert!(BackendKind::parse("opencl").is_err());
        assert!(BackendKind::parse("wgpu").is_err());
    }

    #[test]
    fn resolve_cpu_backend() {
        let device = resolve_device(BackendKind::Cpu).unwrap();
        assert_eq!(device_label(&device), "cpu");
    }

    #[test]
    fn default_backend_when_env_missing() {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        assert_eq!(default_backend_kind(), BackendKind::Cpu);
        #[cfg(any(feature = "cuda", feature = "metal"))]
        assert_eq!(default_backend_kind(), BackendKind::Auto);
    }

    #[test]
    fn resolve_auto_backend_without_gpu_features() {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let device = resolve_device(BackendKind::Auto).unwrap();
            assert_eq!(device_label(&device), "cpu");
        }
    }

    #[test]
    fn resolve_auto_backend_with_gpu_features() {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            let device = resolve_device(BackendKind::Auto).unwrap();
            assert!(matches!(
                device_label(&device),
                "cuda" | "metal" | "cpu"
            ));
        }
    }

    #[test]
    fn gpu_backends_require_cargo_features() {
        #[cfg(not(feature = "cuda"))]
        assert!(resolve_device(BackendKind::Cuda).is_err());
        #[cfg(not(feature = "metal"))]
        assert!(resolve_device(BackendKind::Metal).is_err());
    }
}
