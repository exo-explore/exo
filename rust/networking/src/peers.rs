use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// IPv4 + IPv6 listen locators. Windows IPv6 sockets are v6-only by default, so
/// `[::]` does not accept IPv4 peers; bind `0.0.0.0` as well.
pub fn default_listen_endpoints(listen_port: u16) -> String {
    format!("[\"tcp/[::]:{listen_port}\", \"tcp/0.0.0.0:{listen_port}\"]")
}

pub fn ipv4_only_listen_endpoints(listen_port: u16) -> String {
    format!("[\"tcp/0.0.0.0:{listen_port}\"]")
}

/// Parse `EXO_BOOTSTRAP_PEERS` / `--bootstrap-peers`: comma-separated
/// `host`, `host:port`, or `tcp/host:port`. Missing port uses the local zenoh port.
pub fn parse_bootstrap_peers(raw: &str, default_port: u16) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .filter_map(|part| normalize_tcp_locator(part, default_port))
        .collect()
}

pub fn bootstrap_peers_from_env(default_port: u16) -> Vec<String> {
    std::env::var("EXO_BOOTSTRAP_PEERS")
        .ok()
        .map(|raw| parse_bootstrap_peers(&raw, default_port))
        .unwrap_or_default()
}

pub fn connect_endpoints_json(locators: &[String]) -> String {
    let inner: Vec<String> = locators
        .iter()
        .map(|locator| format!("\"{locator}\""))
        .collect();
    format!("[{}]", inner.join(", "))
}

/// Lower is better. Prefer IPv4 on the same /24 as a local ethernet/thunderbolt NIC.
pub fn locator_preference(addr: SocketAddr, ethernet_v4: &[Ipv4Addr]) -> u8 {
    match addr.ip() {
        IpAddr::V4(v4) if ethernet_v4.iter().any(|local| same_slash24(*local, v4)) => 0,
        IpAddr::V4(v4) if !v4.is_link_local() => 1,
        IpAddr::V4(_) => 2,
        IpAddr::V6(v6) if v6.is_unicast_link_local() => 4,
        IpAddr::V6(_) => 3,
    }
}

pub fn iface_is_wifi(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    ["wi-fi", "wifi", "wlan", "wireless"]
        .iter()
        .any(|token| lowered.contains(token))
}

pub fn iface_is_wired(name: &str) -> bool {
    if iface_is_wifi(name) {
        return false;
    }
    let lowered = name.to_ascii_lowercase();
    ["ethernet", "eth", "lan", "thunderbolt", "local area"]
        .iter()
        .any(|token| lowered.contains(token))
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut acc, byte| {
        acc.push_str(&format!("{byte:02x}"));
        acc
    })
}

pub fn parse_hex_bytes<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for i in 0..N {
        out[i] = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn normalize_tcp_locator(raw: &str, default_port: u16) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix("tcp/") {
        if rest.is_empty() {
            return None;
        }
        return Some(trimmed.to_owned());
    }
    if trimmed.parse::<SocketAddr>().is_ok() {
        return Some(format!("tcp/{trimmed}"));
    }
    if let Some((host, port)) = split_host_port(trimmed) {
        if !host.is_empty() && port.parse::<u16>().is_ok() {
            return Some(format!("tcp/{trimmed}"));
        }
    }
    if trimmed.contains('/') || trimmed.contains(' ') {
        return None;
    }
    Some(format!("tcp/{trimmed}:{default_port}"))
}

fn split_host_port(s: &str) -> Option<(&str, &str)> {
    if let Some(rest) = s.strip_prefix('[') {
        let (host, tail) = rest.split_once(']')?;
        let port = tail.strip_prefix(':')?;
        if host.is_empty() {
            return None;
        }
        return Some((host, port));
    }
    let (host, port) = s.rsplit_once(':')?;
    if host.is_empty() || port.contains(':') {
        return None;
    }
    Some((host, port))
}

fn same_slash24(a: Ipv4Addr, b: Ipv4Addr) -> bool {
    let a = a.octets();
    let b = b.octets();
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_port_and_bare_ip() {
        let peers = parse_bootstrap_peers("192.168.1.10,10.0.0.2:52414, tcp/127.0.0.1:9", 52414);
        assert_eq!(
            peers,
            vec![
                "tcp/192.168.1.10:52414".to_owned(),
                "tcp/10.0.0.2:52414".to_owned(),
                "tcp/127.0.0.1:9".to_owned(),
            ]
        );
    }

    #[test]
    fn prefers_ethernet_subnet() {
        let ethernet = [Ipv4Addr::new(192, 168, 1, 5)];
        let wired = "192.168.1.9:52414".parse().expect("wired fixture");
        let wifi = "10.0.0.9:52414".parse().expect("wifi fixture");
        assert!(locator_preference(wired, &ethernet) < locator_preference(wifi, &ethernet));
    }
}
