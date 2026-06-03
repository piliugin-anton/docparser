#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use docparser_api::{ApiConfig, load_env_file, run};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    load_env_file();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("docparser_api=info".parse()?))
        .init();
    run(ApiConfig::default()).await
}
