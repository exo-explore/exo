use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;

use log::warn;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use tokio::sync::mpsc;
use zenoh::config::ZenohId;

use crate::discovery::Discovered;
use crate::peers::{hex_bytes, parse_hex_bytes};

pub const SERVICE_TYPE: &str = "_exo._tcp.local.";

pub struct MdnsDiscovery {
    _daemon: ServiceDaemon,
    pub rx: mpsc::Receiver<Discovered>,
}

impl MdnsDiscovery {
    pub fn start(zid: ZenohId, namespace: [u8; 8], listen_port: u16) -> io::Result<Self> {
        let daemon = ServiceDaemon::new().map_err(io::Error::other)?;
        let ns_hex = hex_bytes(&namespace);
        let zid_hex = hex_bytes(&zid.to_le_bytes());
        let instance = format!("exo-{}", &zid_hex[..8.min(zid_hex.len())]);
        let host = format!("{}.local.", mdns_hostname());
        let props = [("ns", ns_hex.as_str()), ("zid", zid_hex.as_str())];
        match ServiceInfo::new(SERVICE_TYPE, &instance, &host, "", listen_port, &props[..]) {
            Ok(info) => {
                if let Err(e) = daemon.register(info) {
                    warn!("mDNS register failed: {e}");
                }
            }
            Err(e) => warn!("mDNS service info failed: {e}"),
        }

        let browser = daemon.browse(SERVICE_TYPE).map_err(io::Error::other)?;
        let (tx, rx) = mpsc::channel(32);
        std::thread::Builder::new()
            .name("exo-mdns".to_owned())
            .spawn(move || {
                while let Ok(event) = browser.recv() {
                    let ServiceEvent::ServiceResolved(info) = event else {
                        continue;
                    };
                    if let Some(discovered) = discovered_from_info(&info, namespace, zid) {
                        if tx.blocking_send(discovered).is_err() {
                            break;
                        }
                    }
                }
            })?;

        Ok(Self {
            _daemon: daemon,
            rx,
        })
    }
}

impl Drop for MdnsDiscovery {
    fn drop(&mut self) {
        let _ = self._daemon.shutdown();
    }
}

fn discovered_from_info(
    info: &ServiceInfo,
    namespace: [u8; 8],
    my_zid: ZenohId,
) -> Option<Discovered> {
    let props: HashMap<String, String> = info
        .get_properties()
        .iter()
        .map(|prop| (prop.key().to_owned(), prop.val_str().to_owned()))
        .collect();
    let ns = props.get("ns")?;
    let ns_bytes = parse_hex_bytes::<8>(ns)?;
    if ns_bytes != namespace {
        return None;
    }
    let zid_bytes = parse_hex_bytes::<16>(props.get("zid")?)?;
    let zid = ZenohId::try_from(zid_bytes.as_slice()).ok()?;
    if zid == my_zid {
        return None;
    }
    let port = info.get_port();
    if port == 0 {
        return None;
    }
    let addr = info.get_addresses().iter().copied().find_map(|ip| {
        if ip.is_loopback() || ip.is_unspecified() {
            return None;
        }
        Some(SocketAddr::new(ip, port))
    })?;
    Some(Discovered { zid, addr })
}

fn mdns_hostname() -> String {
    let raw = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "exo-node".to_owned());
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "exo-node".to_owned()
    } else {
        cleaned
    }
}
