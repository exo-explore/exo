use clap::{ArgAction, Parser, ValueEnum};
use serde::{Deserialize, Serialize};

// TODO: need to figure out incremental compilation and somehow include the right version
//       for CLAP so that we display the right version either with shaddow-rs or include!()

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[command(name = "EXO", version, about, long_about = None)]
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
    bootstrap_peers: Option<Vec<String>>,

    #[arg(
        long = "libp2p-port",
        default_value_t = 0,
        value_name = "PORT",
        help = "Fixed TCP port for libp2p to listen on (0 = OS-assigned)"
    )]
    libp2p_port: u16,

    #[arg(
        long = "fast-synch",
        value_name = "BOOL",
        help = "Force MLX FAST_SYNCH on/off (for JACCL backend); omit for auto"
    )]
    fast_synch: Option<bool>,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value.eq_ignore_ascii_case("true"))
}
