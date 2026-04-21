//! Process configuration and shared constants for `babblerd`.
//!
//! This module centralizes:
//!
//! - default runtime paths
//! - environment variable overrides
//! - protocol/application constants such as the mesh prefix
//! - coarse daemon defaults such as the router UDP port

use color_eyre::eyre::{self, eyre};
use ipnet::Ipv6Net;
use std::env;
use std::net::Ipv6Addr;
use std::path::{Path, PathBuf};

pub const PUBLIC_SOCKET_PATH_ENV: &str = "BABBLER_SOCKET_PATH";
pub const NODE_ID_FILE_ENV: &str = "BABBLER_NODE_ID_FILE";
pub const ROUTER_UDP_PORT_ENV: &str = "BABBLER_ROUTER_UDP_PORT";

pub const DEFAULT_PUBLIC_SOCKET_PATH: &str = {
    #[cfg(target_os = "macos")]
    {
        "/var/run/babbler/babblerd.sock"
    }
    #[cfg(target_os = "linux")]
    {
        "/run/babbler/babblerd.sock"
    }
};

pub const DEFAULT_NODE_ID_FILE: &str = {
    #[cfg(target_os = "macos")]
    {
        "/var/db/babbler/node-id"
    }
    #[cfg(target_os = "linux")]
    {
        "/var/lib/babbler/node-id"
    }
};

// TODO: just picked a random one that didn't seem occupied, there is probably a better way
//       to do this in the future :)
pub const DEFAULT_ROUTER_UDP_PORT: u16 = 41897;

pub const EXO_ULA_PREFIX: Ipv6Net = Ipv6Net::new_assert(
    // TODO: break out into "fd" for ULA
    //       e0_20c61fa7 for EXO address-space
    //       ffff for anything else we want, like maybe versioning and so on (but for now its not used)
    Ipv6Addr::from_bits(0xfd_e0_20c61fa7_ffff << 80),
    64,
);

#[derive(Debug, Clone)]
pub struct Config {
    pub public_socket_path: PathBuf,
    pub public_dir: PathBuf,
    pub node_id_file: PathBuf,
    pub router_udp_port: u16,
    pub exo_ula_prefix: Ipv6Net,
}

impl Config {
    pub fn from_env() -> eyre::Result<Self> {
        let public_socket_path = env::var_os(PUBLIC_SOCKET_PATH_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_PUBLIC_SOCKET_PATH));
        let Some(public_dir) = public_socket_path.parent().map(Path::to_path_buf) else {
            return Err(eyre!(
                "public socket path has no parent directory: {}",
                public_socket_path.display()
            ));
        };

        let node_id_file = env::var_os(NODE_ID_FILE_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_NODE_ID_FILE));

        let router_udp_port = match env::var(ROUTER_UDP_PORT_ENV) {
            Ok(raw) => raw
                .parse::<u16>()
                .map_err(|e| eyre!("invalid {ROUTER_UDP_PORT_ENV} value {raw:?}: {e}"))?,
            Err(_) => DEFAULT_ROUTER_UDP_PORT,
        };

        Ok(Self {
            public_socket_path,
            public_dir,
            node_id_file,
            router_udp_port,
            exo_ula_prefix: EXO_ULA_PREFIX,
        })
    }
}
