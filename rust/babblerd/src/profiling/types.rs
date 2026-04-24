use std::net::Ipv6Addr;
use std::time::Duration;

use crate::config::TUN_MTU;

pub const DEFAULT_PROFILE_PORT: u16 = 41_901;
pub const DEFAULT_ECHO_COUNT: u32 = 10;
pub const DEFAULT_ECHO_INTERVAL_MS: u64 = 250;
pub const DEFAULT_ECHO_TIMEOUT_MS: u64 = 500;
pub const DEFAULT_CAPACITY_ROUNDS: u32 = 5;
pub const DEFAULT_TRAIN_PACKETS: u32 = 64;
pub const DEFAULT_TRAIN_INTERVAL_MS: u64 = 1_000;
pub const DEFAULT_TRAIN_SETTLE_MS: u64 = 25;

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
            echo_count: DEFAULT_ECHO_COUNT,
            echo_interval: Duration::from_millis(DEFAULT_ECHO_INTERVAL_MS),
            echo_timeout: Duration::from_millis(DEFAULT_ECHO_TIMEOUT_MS),
            capacity_rounds: DEFAULT_CAPACITY_ROUNDS,
            train_packets: DEFAULT_TRAIN_PACKETS,
            train_payload_bytes: usize::from(TUN_MTU),
            train_interval: Duration::from_millis(DEFAULT_TRAIN_INTERVAL_MS),
            train_settle: Duration::from_millis(DEFAULT_TRAIN_SETTLE_MS),
        }
    }
}
