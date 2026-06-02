use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::manifest;

/// Return true when an existing file matches the expected byte size (skip re-download).
pub fn should_skip(path: &Path, expected_size: Option<u64>) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let meta = path.metadata().context("read metadata")?;
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

pub fn verify_models_dir(models_dir: &Path) -> Result<()> {
    let vlm = models_dir.join(manifest::VLM_DIR_NAME);
    let layout = models_dir.join(manifest::LAYOUT_DIR_NAME);

    for file in manifest::VLM_REQUIRED {
        let p = vlm.join(file);
        if !p.is_file() {
            bail!("missing VLM artifact: {}", p.display());
        }
    }
    for file in manifest::LAYOUT_REQUIRED {
        let p = layout.join(file);
        if !p.is_file() {
            bail!("missing layout artifact: {}", p.display());
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
