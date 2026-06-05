use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::try_join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::error::{DownloadError, Result};
use crate::hf::hf_resolve_url;
use crate::manifest;
use crate::verify::should_skip_async;

pub struct DownloadOptions {
    pub jobs: usize,
    pub dry_run: bool,
    pub hf_token: Option<String>,
}

impl Clone for DownloadOptions {
    fn clone(&self) -> Self {
        Self {
            jobs: self.jobs,
            dry_run: self.dry_run,
            hf_token: self.hf_token.clone(),
        }
    }
}

#[allow(clippy::too_many_arguments)] // mirrors CLI flags; grouping opts would not reduce call-site clarity
pub async fn download_all(
    models_dir: &Path,
    fixtures_dir: &Path,
    include_reference: bool,
    vlm_only: bool,
    layout_only: bool,
    doc_prep_only: bool,
    fixtures_only: bool,
    opts: &DownloadOptions,
) -> Result<()> {
    let mut joins: Vec<JoinHandle<Result<()>>> = Vec::new();
    let full = !vlm_only && !layout_only && !fixtures_only && !doc_prep_only;

    if vlm_only || full {
        let mut vlm_files: Vec<&'static str> = manifest::VLM_REQUIRED.to_vec();
        if include_reference {
            vlm_files.extend(manifest::VLM_REFERENCE);
        }
        let dest = models_dir.join(manifest::VLM_DIR_NAME);
        let opts = opts.clone();
        joins.push(tokio::spawn(async move {
            download_repo(
                manifest::VLM_REPO,
                &dest,
                &vlm_files,
                manifest::VLM_SIZES,
                &opts,
            )
            .await
        }));
    }

    if layout_only || full {
        let dest = models_dir.join(manifest::LAYOUT_DIR_NAME);
        let opts_layout = opts.clone();
        joins.push(tokio::spawn(async move {
            download_repo(
                manifest::LAYOUT_REPO,
                &dest,
                manifest::LAYOUT_REQUIRED,
                manifest::LAYOUT_SIZES,
                &opts_layout,
            )
            .await
        }));
    }

    if doc_prep_only || full {
        let dest = models_dir.join(manifest::DOC_ORI_DIR_NAME);
        let opts_doc = opts.clone();
        joins.push(tokio::spawn(async move {
            download_repo(
                manifest::DOC_ORI_REPO,
                &dest,
                manifest::DOC_ORI_REQUIRED,
                manifest::DOC_ORI_SIZES,
                &opts_doc,
            )
            .await
        }));
        let dest = models_dir.join(manifest::UVDOC_DIR_NAME);
        let opts_uv = opts.clone();
        joins.push(tokio::spawn(async move {
            download_repo(
                manifest::UVDOC_REPO,
                &dest,
                manifest::UVDOC_REQUIRED,
                manifest::UVDOC_SIZES,
                &opts_uv,
            )
            .await
        }));
    }

    if !vlm_only && !layout_only && !doc_prep_only {
        let fixtures_dir = fixtures_dir.to_path_buf();
        let opts = opts.clone();
        joins.push(tokio::spawn(async move {
            download_fixtures(&fixtures_dir, &opts).await
        }));
    }

    for join in joins {
        join.await??;
    }
    Ok(())
}

fn spinner_style() -> Result<ProgressStyle> {
    Ok(ProgressStyle::with_template("{spinner} {msg}")
        .map_err(|e| DownloadError::Message(format!("invalid spinner progress template: {e}")))?
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]))
}

fn bar_style() -> Result<ProgressStyle> {
    Ok(
        ProgressStyle::with_template("{msg} [{bar:40}] {bytes}/{total_bytes} ({eta})")
            .map_err(|e| DownloadError::Message(format!("invalid bar progress template: {e}")))?
            .progress_chars("=>-"),
    )
}

async fn acquire_permit(sem: &Semaphore) -> Result<tokio::sync::SemaphorePermit<'_>> {
    sem.acquire()
        .await
        .map_err(|_| DownloadError::SemaphoreClosed)
}

async fn download_repo(
    repo: &str,
    dest: &Path,
    files: &[&str],
    sizes: &[(&str, u64)],
    opts: &DownloadOptions,
) -> Result<()> {
    if opts.dry_run {
        for file in files {
            let url = hf_resolve_url(repo, file);
            println!("DRY RUN {} -> {}", url, dest.join(file).display());
        }
        return Ok(());
    }

    fs::create_dir_all(dest).await.map_err(|e| {
        DownloadError::Io(std::io::Error::new(
            e.kind(),
            format!("create {}: {e}", dest.display()),
        ))
    })?;

    let client = build_client(opts)?;
    let sem = Arc::new(Semaphore::new(opts.jobs.max(1)));
    let mp = MultiProgress::new();

    let futs: Vec<_> = files
        .iter()
        .map(|file| {
            let client = client.clone();
            let sem = Arc::clone(&sem);
            let dest = dest.to_path_buf();
            let repo = repo.to_string();
            let file = file.to_string();
            let mp = mp.clone();
            let opts = opts.clone();
            let expected = manifest::expected_size(sizes, &file);
            async move {
                let _permit = acquire_permit(&sem).await?;
                download_one(&client, &repo, &dest, &file, expected, &mp, &opts).await
            }
        })
        .collect();

    try_join_all(futs).await?;
    Ok(())
}

async fn download_fixtures(fixtures_dir: &Path, opts: &DownloadOptions) -> Result<()> {
    if opts.dry_run {
        for fx in manifest::FIXTURES {
            println!(
                "DRY RUN {} -> {}",
                fx.url,
                fixtures_dir.join(fx.filename).display()
            );
        }
        return Ok(());
    }

    fs::create_dir_all(fixtures_dir).await?;
    let client = build_client(opts)?;
    let sem = Arc::new(Semaphore::new(opts.jobs.max(1)));
    let mp = MultiProgress::new();

    let futs: Vec<_> = manifest::FIXTURES
        .iter()
        .map(|fx| {
            let client = client.clone();
            let sem = Arc::clone(&sem);
            let opts = opts.clone();
            let dest = fixtures_dir.join(fx.filename);
            let url = fx.url.to_string();
            let name = fx.filename.to_string();
            let mp = mp.clone();
            async move {
                let _permit = acquire_permit(&sem).await?;
                if should_skip_async(&dest, None).await? {
                    tracing::info!("skip (exists): {}", dest.display());
                    return Ok::<(), DownloadError>(());
                }
                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(spinner_style()?);
                pb.set_message(format!("fixture {name}"));
                stream_to_file(&client, &url, &dest, None, Some(&pb), Some(&opts)).await?;
                pb.finish_with_message(format!("done {name}"));
                Ok(())
            }
        })
        .collect();

    try_join_all(futs).await?;
    Ok(())
}

async fn download_one(
    client: &Client,
    repo: &str,
    dest_dir: &Path,
    file: &str,
    expected_size: Option<u64>,
    mp: &MultiProgress,
    opts: &DownloadOptions,
) -> Result<()> {
    let dest = dest_dir.join(file);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await.ok();
    }

    if should_skip_async(&dest, expected_size).await? {
        tracing::info!("skip (size ok): {}", dest.display());
        return Ok(());
    }

    let url = hf_resolve_url(repo, file);
    let pb = if expected_size.unwrap_or(0) > 1_000_000 {
        let bar = mp.add(ProgressBar::new(expected_size.unwrap_or(0)));
        bar.set_style(bar_style()?);
        bar.set_message(file.to_string());
        Some(bar)
    } else {
        let bar = mp.add(ProgressBar::new_spinner());
        bar.set_style(spinner_style()?);
        bar.set_message(file.to_string());
        Some(bar)
    };

    stream_to_file(client, &url, &dest, expected_size, pb.as_ref(), Some(opts)).await?;

    if let Some(pb) = pb {
        pb.finish_with_message(format!("done {file}"));
    }
    Ok(())
}

/// Temp path beside `dest` that keeps the full basename (e.g. `tokenizer.json.part`).
/// `dest.with_extension("part")` would collapse `tokenizer.json` and `tokenizer.model`
/// both to `tokenizer.part` and break parallel downloads.
fn partial_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "download".into());
    dest.with_file_name(format!("{name}.part"))
}

async fn stream_to_file(
    client: &Client,
    url: &str,
    dest: &Path,
    total_hint: Option<u64>,
    pb: Option<&ProgressBar>,
    opts: Option<&DownloadOptions>,
) -> Result<()> {
    let part = partial_path(dest);
    let mut req = client.get(url);
    let env_token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    if let Some(token) = opts
        .and_then(|o| o.hf_token.as_deref())
        .or(env_token.as_deref())
    {
        req = req.bearer_auth(token);
    }

    let resp = req
        .send()
        .await
        .map_err(DownloadError::Http)?
        .error_for_status()
        .map_err(DownloadError::Http)?;

    let total = resp.content_length().or(total_hint);
    if let (Some(pb), Some(total)) = (pb, total) {
        pb.set_length(total);
    }

    let mut file = fs::File::create(&part).await.map_err(|e| {
        DownloadError::Io(std::io::Error::new(
            e.kind(),
            format!("create {}: {e}", part.display()),
        ))
    })?;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(DownloadError::Http)?;
        file.write_all(&chunk).await.map_err(DownloadError::Io)?;
        if let Some(pb) = pb {
            pb.inc(chunk.len() as u64);
        }
    }
    file.flush().await.map_err(DownloadError::Io)?;
    drop(file);
    fs::rename(&part, dest).await.map_err(|e| {
        DownloadError::Io(std::io::Error::new(
            e.kind(),
            format!("rename {} -> {}: {e}", part.display(), dest.display()),
        ))
    })?;
    Ok(())
}

fn build_client(opts: &DownloadOptions) -> Result<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(ref token) = opts.hf_token {
        let auth = format!("Bearer {token}").parse().map_err(|e| {
            DownloadError::Message(format!("invalid HF bearer token header value: {e}"))
        })?;
        headers.insert(reqwest::header::AUTHORIZATION, auth);
    }
    Client::builder()
        .user_agent("docparser-download/0.1")
        .default_headers(headers)
        .build()
        .map_err(DownloadError::Http)
}

pub fn default_models_dir() -> PathBuf {
    PathBuf::from("models")
}

pub fn default_fixtures_dir() -> PathBuf {
    PathBuf::from("tests/fixtures")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_path_keeps_distinct_basenames() {
        let dir = Path::new("models/PaddleOCR-VL-1.6");
        let json = partial_path(&dir.join("tokenizer.json"));
        let model = partial_path(&dir.join("tokenizer.model"));
        assert_eq!(json, dir.join("tokenizer.json.part"));
        assert_eq!(model, dir.join("tokenizer.model.part"));
        assert_ne!(json, model);
    }
}
