use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use docparser_download::{DownloadOptions, download_all, verify_models_dir};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "docparser-download")]
#[command(about = "Download HuggingFace model artifacts and parity fixtures in parallel")]
struct Args {
    /// Directory for VLM and layout model trees
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Directory for test fixture images
    #[arg(long, default_value = "tests/fixtures")]
    fixtures_dir: PathBuf,

    /// Concurrent file downloads per repo
    #[arg(long, default_value = "8")]
    jobs: usize,

    /// Also fetch VLM reference *.py files from HuggingFace
    #[arg(long)]
    include_reference: bool,

    #[arg(long)]
    vlm_only: bool,

    #[arg(long)]
    layout_only: bool,

    #[arg(long)]
    doc_prep_only: bool,

    #[arg(long)]
    fixtures_only: bool,

    #[arg(long)]
    dry_run: bool,

    /// Verify artifacts exist after download (no network)
    #[arg(long)]
    verify_only: bool,
}

async fn verify_models_dir_blocking(models_dir: &Path) -> Result<()> {
    let models_dir = models_dir.to_path_buf();
    tokio::task::spawn_blocking(move || verify_models_dir(&models_dir))
        .await
        .context("verify task join error")?
        .context("model verification failed")
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("docparser_download=info".parse()?),
        )
        .init();

    let args = Args::parse();

    if args.verify_only {
        verify_models_dir_blocking(&args.models_dir).await?;
        println!(
            "All required model artifacts present under {}",
            args.models_dir.display()
        );
        return Ok(());
    }

    let hf_token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    let opts = DownloadOptions {
        jobs: args.jobs,
        dry_run: args.dry_run,
        hf_token,
    };

    download_all(
        &args.models_dir,
        &args.fixtures_dir,
        args.include_reference,
        args.vlm_only,
        args.layout_only,
        args.doc_prep_only,
        args.fixtures_only,
        &opts,
    )
    .await
    .context("download failed")?;

    if !args.dry_run {
        verify_models_dir_blocking(&args.models_dir).await?;
        println!(
            "Download complete.\n  models: {}\n  fixtures: {}",
            args.models_dir.display(),
            args.fixtures_dir.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use std::path::PathBuf;

    use docparser_download::{default_fixtures_dir, default_models_dir};

    #[test]
    fn default_paths() {
        assert_eq!(default_models_dir(), PathBuf::from("models"));
        assert_eq!(default_fixtures_dir(), PathBuf::from("tests/fixtures"));
    }
}
