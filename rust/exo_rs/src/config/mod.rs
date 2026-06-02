use clap::{ArgAction, Parser};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub verbosity: i32,
    pub force_master: bool,
    pub spawn_api: bool,
    pub api_port: u16,
    pub tb_only: bool,
    pub no_worker: bool,
    pub no_downloads: bool,
    pub offline: bool,
    pub no_batch: bool,
    pub fast_synch: Option<bool>,
    pub legacy_daemon: bool,
    pub bootstrap_peers: Vec<String>,
    pub libp2p_port: u16,
}

impl Args {
    pub fn parse() -> Self {
        CliArgs::parse().into()
    }
}

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[command(name = "EXO", version, about, long_about = None)]
struct CliArgs {
    #[arg(short = 'q', long = "quiet", action = ArgAction::SetTrue)]
    quiet: bool,

    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    verbose: u8,

    #[arg(short = 'm', long = "force-master", action = ArgAction::SetTrue)]
    force_master: bool,

    #[arg(long = "no-api", action = ArgAction::SetFalse, default_value_t = true)]
    spawn_api: bool,

    #[arg(long = "api-port", default_value_t = 52415, value_parser = parse_positive_port)]
    api_port: u16,

    #[arg(long = "no-worker", action = ArgAction::SetTrue)]
    no_worker: bool,

    #[arg(
        long = "no-downloads",
        action = ArgAction::SetTrue,
        help = "Disable the download coordinator (node won't download models)"
    )]
    no_downloads: bool,

    #[arg(
        long = "offline",
        action = ArgAction::SetTrue,
        help = "Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models"
    )]
    offline: bool,

    #[arg(
        long = "no-batch",
        action = ArgAction::SetTrue,
        help = "Disable continuous batching, use sequential generation"
    )]
    no_batch: bool,

    #[arg(
        long = "legacy-daemon",
        action = ArgAction::SetTrue,
        help = "Run as a legacy SysV-style background daemon using double-fork daemonization"
    )]
    legacy_daemon: bool,

    #[arg(
        long = "bootstrap-peers",
        value_name = "PEERS",
        help = "Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)"
    )]
    bootstrap_peers: Option<String>,

    #[arg(
        long = "libp2p-port",
        default_value_t = 0,
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

impl From<CliArgs> for Args {
    fn from(cli_args: CliArgs) -> Self {
        Self {
            verbosity: verbosity(cli_args.quiet, cli_args.verbose),
            force_master: cli_args.force_master,
            spawn_api: cli_args.spawn_api,
            api_port: cli_args.api_port,
            tb_only: false,
            no_worker: cli_args.no_worker,
            no_downloads: cli_args.no_downloads,
            offline: cli_args.offline || env_flag("EXO_OFFLINE"),
            no_batch: cli_args.no_batch,
            fast_synch: cli_args.fast_synch,
            legacy_daemon: cli_args.legacy_daemon,
            bootstrap_peers: bootstrap_peers(cli_args.bootstrap_peers),
            libp2p_port: cli_args.libp2p_port,
        }
    }
}

fn verbosity(quiet: bool, verbose: u8) -> i32 {
    if quiet { -1 } else { i32::from(verbose) }
}

fn bootstrap_peers(argument: Option<String>) -> Vec<String> {
    match argument {
        Some(peers) => split_comma_separated(&peers),
        None => std::env::var("EXO_BOOTSTRAP_PEERS")
            .map(|peers| split_comma_separated(&peers))
            .unwrap_or_default(),
    }
}

fn split_comma_separated(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter(|peer| !peer.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value.eq_ignore_ascii_case("true"))
}

fn parse_positive_port(value: &str) -> Result<u16, String> {
    let port = value
        .parse::<u16>()
        .map_err(|error| format!("expected a positive port number: {error}"))?;

    if port == 0 {
        Err("expected a positive port number".to_owned())
    } else {
        Ok(port)
    }
}
