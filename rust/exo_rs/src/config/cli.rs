use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

pub const EXO_VERSION: &str = match option_env!("EXO_PKG_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Parser)]
#[command(name = "EXO", version = EXO_VERSION, about, long_about = None)]
pub struct CliArgs {
    #[arg(
          short = 'v',
          long = "verbosity",
          value_enum,
          default_value_t = Verbosity::Info,
          value_name = "LEVEL",
          help = "Set the verbosity level"
    )]
    pub verbosity: Verbosity,

    #[arg(
        short = 'm',
        long = "force-master",
        action = ArgAction::SetTrue,
        help = "Force node to be master"
    )]
    force_master: bool,

    #[arg(
        long = "no-api",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the API"
    )]
    api_enabled: bool,

    #[arg(
        long = "api-port",
        default_value_t = 52415,
        value_name = "PORT",
        help = "Port on which the API runs"
    )]
    api_port: u16,

    #[arg(
        long = "no-worker",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the worker"
    )]
    worker_enabled: bool,

    #[arg(
        long = "no-downloads",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable the download coordinator (node won't download models)"
    )]
    downloads_enabled: bool,

    #[arg(
        long = "offline",
        action = ArgAction::SetTrue,
        help = "Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models"
    )]
    offline: bool,

    #[arg(
        long = "no-batch",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Disable continuous batching, use sequential generation"
    )]
    continuous_batching_enabled: bool,

    #[arg(
        long = "legacy-daemon",
        action = ArgAction::SetTrue,
        help = "Run as a legacy SysV-style background daemon using double-fork daemonization"
    )]
    legacy_daemon: bool,

    #[arg(
        long = "bootstrap-peers",
        value_delimiter = ',',
        value_name = "MULTIADDRS",
        help = "Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)"
    )]
    #[deprecated]
    bootstrap_peers: Option<Vec<String>>,

    #[arg(
        long,
        default_value_t = EXO_VERSION.to_string(),
        value_name = "STRING",
        help = "Discovery namespace, nodes with different namespaces will not connect."
    )]
    namespace: String,

    #[arg(
        long = "zenoh-port",
        default_value_t = 52414,
        value_name = "PORT",
        help = "Fixed TCP port for zenoh to listen."
    )]
    zenoh_port: u16,

    #[arg(
        long = "discovery-port",
        default_value_t = 52413,
        value_name = "PORT",
        help = "Fixed UDP port for the discovery service."
    )]
    discovery_port: u16,

    #[arg(
        long = "fast-synch",
        value_name = "BOOL",
        help = "Force MLX FAST_SYNCH on/off (for JACCL backend); omit for auto"
    )]
    fast_synch: Option<bool>,

    #[command(flatten)]
    deprecated: DeprecatedArgs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, clap::Args)]
pub struct DeprecatedArgs {
    #[arg(long = "libp2p-port", hide = true)]
    libp2p_port: Option<u16>,
}

impl DeprecatedArgs {
    pub fn get_error(&self) -> Option<clap::Error> {
        // destructure: don't change because this errors when new options are moved here
        let Self { libp2p_port } = self.clone();

        if let Some(_) = libp2p_port {
            Some(clap::Error::raw(
                clap::error::ErrorKind::UnknownArgument,
                "The argument --libp2p-port is deprecated; use --zenoh-port instead",
            ))
        } else {
            None
        }
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value.eq_ignore_ascii_case("true"))
}
