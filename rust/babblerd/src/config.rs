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
use std::collections::HashSet;
use std::env;
use std::fmt::{Display, Formatter};
use std::net::Ipv6Addr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub const PUBLIC_SOCKET_PATH_ENV: &str = "BABBLER_SOCKET_PATH";
pub const NODE_ID_FILE_ENV: &str = "BABBLER_NODE_ID_FILE";
pub const ROUTER_UDP_PORT_ENV: &str = "BABBLER_ROUTER_UDP_PORT";
pub const ROUTER_TRANSPORT_ENV: &str = "BABBLER_ROUTER_TRANSPORT";
pub const TUN_MTU_ENV: &str = "BABBLER_TUN_MTU";
pub const INTERFACE_ALLOWLIST_ENV: &str = "BABBLER_INTERFACE_ALLOWLIST";

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

pub const PHYSICAL_LINK_MTU: u16 = 1500;
pub const OUTER_IPV6_HEADER_BYTES: u16 = 40;
pub const OUTER_UDP_HEADER_BYTES: u16 = 8;
pub const UDP_TUN_MTU: u16 = PHYSICAL_LINK_MTU - OUTER_IPV6_HEADER_BYTES - OUTER_UDP_HEADER_BYTES;
pub const TCP_TUN_MTU: u16 = 9000;
pub const MIN_TUN_MTU: u16 = 1280;
pub const MAX_TUN_MTU: u16 = u16::MAX;
pub const TUN_MTU: u16 = UDP_TUN_MTU;

pub const EXO_ULA_PREFIX: Ipv6Net = Ipv6Net::new_assert(
    // TODO: break out into "fd" for ULA
    //       e0_20c61fa7 for EXO address-space
    //       ffff for anything else we want, like maybe versioning and so on (but for now its not used)
    //
    // NOTE: spell the hextets explicitly here. A previous `u128` bit-shift
    // construction accidentally truncated the leading `fde0` and produced
    // `20c6:1fa7:ffff::/64`, which is not ULA.
    Ipv6Addr::new(0xfde0, 0x20c6, 0x1fa7, 0xffff, 0, 0, 0, 0),
    64,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Udp,
    Tcp,
}

impl Display for TransportMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Udp => write!(f, "udp"),
            Self::Tcp => write!(f, "tcp"),
        }
    }
}

impl FromStr for TransportMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "udp" => Ok(Self::Udp),
            "tcp" => Ok(Self::Tcp),
            other => Err(format!("expected udp or tcp, got {other:?}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub public_socket_path: PathBuf,
    pub public_dir: PathBuf,
    pub node_id_file: PathBuf,
    pub router_udp_port: u16,
    pub router_transport: TransportMode,
    pub tun_mtu: u16,
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

        let router_transport = match env::var(ROUTER_TRANSPORT_ENV) {
            Ok(raw) => raw
                .parse::<TransportMode>()
                .map_err(|e| eyre!("invalid {ROUTER_TRANSPORT_ENV} value {raw:?}: {e}"))?,
            Err(_) => TransportMode::Udp,
        };
        let tun_mtu = match env::var(TUN_MTU_ENV) {
            Ok(raw) => parse_tun_mtu(&raw)
                .map_err(|e| eyre!("invalid {TUN_MTU_ENV} value {raw:?}: {e}"))?,
            Err(_) => default_tun_mtu(router_transport),
        };

        Ok(Self {
            public_socket_path,
            public_dir,
            node_id_file,
            router_udp_port,
            router_transport,
            tun_mtu,
            exo_ula_prefix: EXO_ULA_PREFIX,
        })
    }
}

pub fn default_tun_mtu(transport: TransportMode) -> u16 {
    match transport {
        TransportMode::Udp => UDP_TUN_MTU,
        TransportMode::Tcp => TCP_TUN_MTU,
    }
}

pub fn parse_tun_mtu(raw: &str) -> Result<u16, String> {
    let mtu = raw
        .trim()
        .parse::<u16>()
        .map_err(|err| format!("expected integer MTU: {err}"))?;
    if !(MIN_TUN_MTU..=MAX_TUN_MTU).contains(&mtu) {
        return Err(format!(
            "expected MTU in {MIN_TUN_MTU}..={MAX_TUN_MTU}, got {mtu}"
        ));
    }
    Ok(mtu)
}

pub fn interface_allowlist_from_env() -> eyre::Result<Option<HashSet<Box<str>>>> {
    let Ok(raw) = env::var(INTERFACE_ALLOWLIST_ENV) else {
        return Ok(None);
    };

    let allowlist = raw
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| name.into())
        .collect::<HashSet<Box<str>>>();

    if allowlist.is_empty() {
        return Err(eyre!(
            "{INTERFACE_ALLOWLIST_ENV} was set but contained no interface names"
        ));
    }

    Ok(Some(allowlist))
}

#[cfg(test)]
mod tests {
    use super::{
        EXO_ULA_PREFIX, TCP_TUN_MTU, TransportMode, UDP_TUN_MTU, default_tun_mtu, parse_tun_mtu,
    };
    use std::net::Ipv6Addr;

    #[test]
    fn exo_ula_prefix_keeps_fde0_high_bits() {
        assert_eq!(
            EXO_ULA_PREFIX.addr(),
            Ipv6Addr::new(0xfde0, 0x20c6, 0x1fa7, 0xffff, 0, 0, 0, 0)
        );
        assert_eq!(EXO_ULA_PREFIX.prefix_len(), 64);
    }

    #[test]
    fn transport_mode_parses_udp_and_tcp() {
        assert_eq!("udp".parse::<TransportMode>().unwrap(), TransportMode::Udp);
        assert_eq!("tcp".parse::<TransportMode>().unwrap(), TransportMode::Tcp);
        assert_eq!("TCP".parse::<TransportMode>().unwrap(), TransportMode::Tcp);
        assert!("quic".parse::<TransportMode>().is_err());
    }

    #[test]
    fn tun_mtu_defaults_follow_transport_and_validate_bounds() {
        assert_eq!(default_tun_mtu(TransportMode::Udp), UDP_TUN_MTU);
        assert_eq!(default_tun_mtu(TransportMode::Tcp), TCP_TUN_MTU);
        assert_eq!(parse_tun_mtu("9000").unwrap(), 9000);
        assert!(parse_tun_mtu("1279").is_err());
        assert!(parse_tun_mtu("65536").is_err());
    }
}
