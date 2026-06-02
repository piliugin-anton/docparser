use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::{BoxFuture, try_join_all};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

use crate::hf::hf_resolve_url;
use crate::manifest;
use crate::verify::should_skip;

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

pub async fn download_all(
    models_dir: &Path,
    fixtures_dir: &Path,
    include_reference: bool,
    vlm_only: bool,
    layout_only: bool,
    fixtures_only: bool,
    opts: &DownloadOptions,
) -> Result<()> {
    let mut tasks: Vec<BoxFuture<'static, Result<()>>> = Vec::new();

    if !layout_only && !fixtures_only {
        let mut vlm_files: Vec<&'static str> = manifest::VLM_REQUIRED.to_vec();
        if include_reference {
            vlm_files.extend(manifest::VLM_REFERENCE);
        }
        let dest = models_dir.join(manifest::VLM_DIR_NAME);
        let opts = opts.clone();
        tasks.push(Box::pin(async move {
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

    if !vlm_only && !fixtures_only {
        let dest = models_dir.join(manifest::LAYOUT_DIR_NAME);
        let opts = opts.clone();
        tasks.push(Box::pin(async move {
            download_repo(
                manifest::LAYOUT_REPO,
                &dest,
                manifest::LAYOUT_REQUIRED,
                manifest::LAYOUT_SIZES,
                &opts,
            )
            .await
        }));
    }

    if !vlm_only && !layout_only {
        let fixtures_dir = fixtures_dir.to_path_buf();
        let opts = opts.clone();
        tasks.push(Box::pin(async move {
            download_fixtures(&fixtures_dir, &opts).await
        }));
    }

    try_join_all(tasks).await?;
    Ok(())
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

    fs::create_dir_all(dest)
        .await
        .with_context(|| format!("create {}", dest.display()))?;

    let client = build_client(opts)?;
    let sem = Arc::new(Semaphore::new(opts.jobs.max(1)));
    let mp = MultiProgress::new();

    let futs: Vec<_> = files
        .iter()
        .map(|file| {
            let client = client.clone();
            let sem = sem.clone();
            let dest = dest.to_path_buf();
            let repo = repo.to_string();
            let file = file.to_string();
            let mp = mp.clone();
            let expected = manifest::expected_size(sizes, &file);
            async move {
                let _permit = sem.acquire().await.unwrap();
                download_one(&client, &repo, &dest, &file, expected, &mp).await
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
            let sem = sem.clone();
            let dest = fixtures_dir.join(fx.filename);
            let url = fx.url.to_string();
            let name = fx.filename.to_string();
            let mp = mp.clone();
            async move {
                let _permit = sem.acquire().await.unwrap();
                if should_skip(&dest, None)? {
                    tracing::info!("skip (exists): {}", dest.display());
                    return Ok::<(), anyhow::Error>(());
                }
                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(
                    ProgressStyle::with_template("{spinner} {msg}")
                        .unwrap()
                        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
                );
                pb.set_message(format!("fixture {name}"));
                stream_to_file(&client, &url, &dest, None, Some(&pb)).await?;
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
) -> Result<()> {
    let dest = dest_dir.join(file);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await.ok();
    }

    if should_skip(&dest, expected_size)? {
        tracing::info!("skip (size ok): {}", dest.display());
        return Ok(());
    }

    let url = hf_resolve_url(repo, file);
    let pb = if expected_size.unwrap_or(0) > 1_000_000 {
        let bar = mp.add(ProgressBar::new(expected_size.unwrap_or(0)));
        bar.set_style(
            ProgressStyle::with_template("{msg} [{bar:40}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );
        bar.set_message(format!("{file}"));
        Some(bar)
    } else {
        let bar = mp.add(ProgressBar::new_spinner());
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        bar.set_message(format!("{file}"));
        Some(bar)
    };

    stream_to_file(client, &url, &dest, expected_size, pb.as_ref()).await?;

    if let Some(pb) = pb {
        pb.finish_with_message(format!("done {file}"));
    }
    Ok(())
}

async fn stream_to_file(
    client: &Client,
    url: &str,
    dest: &Path,
    total_hint: Option<u64>,
    pb: Option<&ProgressBar>,
) -> Result<()> {
    let part = dest.with_extension("part");
    let mut req = client.get(url);
    if let Some(token) = std::env::var("HF_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
    {
        req = req.bearer_auth(token);
    }

    let resp = req
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let total = resp.content_length().or(total_hint);
    if let (Some(pb), Some(total)) = (pb, total) {
        pb.set_length(total);
    }

    let mut file = fs::File::create(&part)
        .await
        .with_context(|| format!("create {}", part.display()))?;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read chunk")?;
        file.write_all(&chunk).await.context("write chunk")?;
        if let Some(pb) = pb {
            pb.inc(chunk.len() as u64);
        }
    }
    file.flush().await?;
    drop(file);
    fs::rename(&part, dest)
        .await
        .with_context(|| format!("rename {} -> {}", part.display(), dest.display()))?;
    Ok(())
}

fn build_client(opts: &DownloadOptions) -> Result<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(ref token) = opts.hf_token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
    }
    Client::builder()
        .user_agent("docparser-download/0.1")
        .default_headers(headers)
        .build()
        .context("build HTTP client")
}

pub fn default_models_dir() -> PathBuf {
    PathBuf::from("models")
}

pub fn default_fixtures_dir() -> PathBuf {
    PathBuf::from("tests/fixtures")
}
