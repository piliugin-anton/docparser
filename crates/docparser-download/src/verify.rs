use std::path::Path;

use tokio::fs;

use crate::error::{VerifyError, VerifyResult};
use crate::manifest;

fn should_skip_from_metadata(
    meta: &std::fs::Metadata,
    expected_size: Option<u64>,
    path: &Path,
) -> VerifyResult<bool> {
    if !meta.is_file() {
        return Ok(false);
    }
    let len = meta.len();
    match expected_size {
        Some(expected) if len == expected => Ok(true),
        Some(expected) => {
            tracing::warn!(
                "size mismatch for {}: have {len}, expected {expected}; re-downloading",
                path.display()
            );
            Ok(false)
        }
        None => Ok(true),
    }
}

/// Return true when an existing file matches the expected byte size (skip re-download).
pub async fn should_skip_async(path: &Path, expected_size: Option<u64>) -> VerifyResult<bool> {
    if !fs::try_exists(path).await? {
        return Ok(false);
    }
    let meta = fs::metadata(path).await?;
    should_skip_from_metadata(&meta, expected_size, path)
}

/// Sync variant for unit tests and blocking callers (e.g. API model load).
pub fn should_skip(path: &Path, expected_size: Option<u64>) -> VerifyResult<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let meta = path.metadata()?;
    should_skip_from_metadata(&meta, expected_size, path)
}

pub fn verify_models_dir(models_dir: &Path) -> VerifyResult<()> {
    let vlm = models_dir.join(manifest::VLM_DIR_NAME);
    let layout = models_dir.join(manifest::LAYOUT_DIR_NAME);

    for file in manifest::VLM_REQUIRED {
        let p = vlm.join(file);
        if !p.is_file() {
            return Err(VerifyError::MissingArtifact {
                kind: "VLM",
                path: p.display().to_string(),
            });
        }
    }
    for file in manifest::LAYOUT_REQUIRED {
        let p = layout.join(file);
        if !p.is_file() {
            return Err(VerifyError::MissingArtifact {
                kind: "layout",
                path: p.display().to_string(),
            });
        }
    }
    let doc_ori = models_dir.join(manifest::DOC_ORI_DIR_NAME);
    for file in manifest::DOC_ORI_REQUIRED {
        let p = doc_ori.join(file);
        if !p.is_file() {
            return Err(VerifyError::MissingArtifact {
                kind: "doc orientation",
                path: p.display().to_string(),
            });
        }
    }
    let uvdoc = models_dir.join(manifest::UVDOC_DIR_NAME);
    for file in manifest::UVDOC_REQUIRED {
        let p = uvdoc.join(file);
        if !p.is_file() {
            return Err(VerifyError::MissingArtifact {
                kind: "UVDoc",
                path: p.display().to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn skip_when_size_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"{}").unwrap();
        assert!(should_skip(&path, Some(2)).unwrap());
        assert!(!should_skip(&path, Some(3)).unwrap());
    }
}
