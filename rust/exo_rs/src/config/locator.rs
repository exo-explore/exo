use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Arguments that are needed to resolve paths to files go here.
///
/// This is needed for such things as resolving the path of the configuration `.toml` file,
/// therefore any args here cannot be specified by the configuration `.toml` file.
///
/// By default, any path-like argument goes here, but can be moved to [`ConfigArgs`] if needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
pub struct LocatorArgs {
    #[arg(long, env, help = "Path to Exo's home directory relative to ~/")]
    pub exo_home: Option<PathBuf>,
    #[arg(long, env, help = "Path to Exo's config file")]
    pub exo_config: Option<PathBuf>,
}
