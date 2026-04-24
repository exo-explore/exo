use std::net::Ipv6Addr;
use std::time::Duration;

use crate::config::TUN_MTU;

pub const DEFAULT_PROFILE_PORT: u16 = 41_901;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LinkKey {
    pub ifname: Box<str>,
    pub ifindex: u32,
    pub peer_link_local: Ipv6Addr,
}

#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub port: u16,
    pub echo_count: u32,
    pub echo_interval: Duration,
    pub echo_timeout: Duration,
    pub capacity_rounds: u32,
    pub train_packets: u32,
    pub train_payload_bytes: usize,
    pub train_interval: Duration,
    pub train_settle: Duration,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_PROFILE_PORT,
            echo_count: 10,
            echo_interval: Duration::from_millis(250),
            echo_timeout: Duration::from_millis(500),
            capacity_rounds: 5,
            train_packets: 64,
            train_payload_bytes: usize::from(TUN_MTU),
            train_interval: Duration::from_secs(1),
            train_settle: Duration::from_millis(25),
        }
    }
}
