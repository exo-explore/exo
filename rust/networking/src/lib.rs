use std::sync::Arc;

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;
use crate::wire::{ClusterKey, OtherVersions};

pub mod discovery;
pub mod membership;
pub mod swarm;
pub mod wire;

pub fn is_valid_zid(identity: &str) -> bool {
    let mut iter = identity.chars();
    iter.next()
        .is_some_and(|c| ('1'..='9').contains(&c) || ('a'..='f').contains(&c))
        && iter.all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
        && identity.len() <= 32
}

pub fn cfg(identity: &str, listen_port: u16) -> Result<zenoh::Config> {
    assert!(is_valid_zid(identity));
    assert!(identity.len() <= 32);
    assert!(listen_port != 0, "must used defined listen port");
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    // Peer discovery relies on IPv6 link-local multicast, which some networks
    // (notably Wi-Fi access points isolating wireless clients) do not forward.
    // EXO_ZENOH_CONNECT lets the operator dial known peers over unicast instead,
    // as a comma-separated list of zenoh endpoints, e.g. "tcp/192.168.1.2:52414".
    if let Ok(peers) = std::env::var("EXO_ZENOH_CONNECT") {
        let endpoints: Vec<String> = peers
            .split(',')
            .filter(|peer| !peer.is_empty())
            .map(|peer| format!("\"{peer}\""))
            .collect();
        if !endpoints.is_empty() {
            cfg.insert_json5("connect/endpoints", &format!("[{}]", endpoints.join(",")))?;
        }
    }
    cfg.insert_json5("adminspace/enabled", "true")?;
    //cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    // How long a silent connection is kept (see `membership::LEASE`)
    cfg.insert_json5(
        "transport/link/tx/lease",
        &membership::LEASE.as_millis().to_string(),
    )?;
    //cfg.insert_json5("timestamping/enabled", "true")?;
    Ok(cfg)
}

/// Opens this node's session. It connects only to nodes with the same `namespace`,
/// `wire_version` (exo's WIRE_VERSION) and `build_id`: see `wire`. Exos from before wire
/// versions are recognised under `namespace`, and under `pre_gate_namespace` if given.
pub async fn open(
    cfg: zenoh::Config,
    namespace: &str,
    wire_version: u16,
    build_id: &str,
    pre_gate_namespace: Option<&str>,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let cluster = ClusterKey::new(namespace, wire_version, build_id, pre_gate_namespace);
    let other_versions = OtherVersions::default();
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let mut discovery = Discovery::new(
        z.zid(),
        cluster,
        other_versions.clone(),
        listen_port,
        discovery_service_port,
    )
    .await?;
    let _jh = Arc::new(AbortOnDrop(tokio::task::spawn(async move {
        loop {
            let Ok(discovered) = discovery.next().await.inspect_err(|e| {
                log::warn!("discovery error {e}");
            }) else {
                continue;
            };

            if discovered.zid > runtime.zid() {
                log::debug!("not connecting to peer with greater zid");
                continue;
            }

            let Ok(locator) =
                Locator::new("tcp", discovered.addr.to_string(), "").inspect_err(|e| {
                    log::warn!("failed to parse locator from addr: {e}");
                })
            else {
                continue;
            };

            // Don't wait for the connection here: a slow dial would stall discovery
            // (and look like this node stalling). zenoh ignores dials to a peer it is
            // already connecting to.
            let runtime = runtime.clone();
            tokio::task::spawn(async move {
                runtime
                    .connect_peer(&discovered.zid.into(), &[locator])
                    .await;
            });
        }
    })));
    Ok(Session {
        z,
        other_versions,
        _jh,
    })
}

struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Clone)]
pub struct Session {
    pub z: ZSession,
    /// Devices discovery hears in this namespace that can't join: another wire version or
    /// build, or an exo from before wire versions.
    pub other_versions: OtherVersions,
    _jh: Arc<AbortOnDrop>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lease_is_the_one_membership_times_from() {
        let cfg = cfg("1f", 52414).expect("config");
        // The setting zenoh's transport reads its lease from
        let lease = *cfg.transport().link().tx().lease();
        assert_eq!(
            u128::from(lease),
            membership::LEASE.as_millis(),
            "zenoh's lease, in ms"
        );
    }

    /// exo uses zenoh for pub/sub and liveliness only. A plugin marked required that fails to
    /// start makes zenoh panic, taking the whole node down, so none is configured.
    #[test]
    fn the_session_config_has_no_plugins() {
        let config = cfg("1234abcd", 52414).expect("a valid config");
        let plugins = config.get_json("plugins").expect("plugins is a config key");
        assert_eq!(plugins, "{}");
    }
}
