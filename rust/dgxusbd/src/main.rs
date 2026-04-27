#[cfg(not(target_os = "linux"))]
compile_error!("dgxusbd is linux-only");

use color_eyre::eyre::{self};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    Ok(())
}
