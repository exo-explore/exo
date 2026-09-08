use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;
use crate::peers::{
    bootstrap_peers_from_env, connect_endpoints_json, default_listen_endpoints,
    ipv4_only_listen_endpoints, locator_preference,
};

pub mod discovery;
pub mod mdns;
pub mod peers;
pub mod swarm;

pub fn is_valid_zid(identity: &str) -> bool {
    let mut iter = identity.chars();
    iter.next()
        .is_some_and(|c| ('1'..='9').contains(&c) || ('a'..='f').contains(&c))
        && iter.all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
        && identity.len() <= 32
}

pub fn cfg(identity: &str, listen_port: u16) -> Result<zenoh::Config> {
    cfg_with_listen(
        identity,
        listen_port,
        &default_listen_endpoints(listen_port),
    )
}

pub fn cfg_with_listen(
    identity: &str,
    listen_port: u16,
    listen_endpoints: &str,
) -> Result<zenoh::Config> {
    assert!(is_valid_zid(identity));
    assert!(identity.len() <= 32);
    assert!(listen_port != 0, "must used defined listen port");
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", listen_endpoints)?;
    apply_bootstrap_peers(&mut cfg, listen_port)?;
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    cfg.insert_json5("adminspace/enabled", "true")?;
    if std::env::var("EXO_ZENOH_JUMBO").ok().as_deref() == Some("1") {
        cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    }
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    //cfg.insert_json5("timestamping/enabled", "true")?;
    cfg.insert_json5("plugins/storage_manager/__required__", "true")?;
    cfg.insert_json5(
        "plugins/storage_manager/storages/mem1",
        r#"{
            key_expr: "storage/mem1/**",
            strip_prefix: "storage/mem1",
            volume: "memory",
            replication: {
                interval: 2,
            }
        }"#,
    )?;
    Ok(cfg)
}

fn apply_bootstrap_peers(cfg: &mut zenoh::Config, listen_port: u16) -> Result<()> {
    let peers = bootstrap_peers_from_env(listen_port);
    if peers.is_empty() {
        return Ok(());
    }
    log::info!("connecting bootstrap peers: {peers:?}");
    cfg.insert_json5("connect/endpoints", &connect_endpoints_json(&peers))?;
    Ok(())
}

pub async fn open(
    cfg: zenoh::Config,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let namespace: [u8; 8] = {
        blake3::hash(namespace.as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("8 is equal to 8")
    };
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let mut discovery =
        Discovery::new(z.zid(), namespace, listen_port, discovery_service_port).await?;
    let _jh = Arc::new(AbortOnDrop(tokio::task::spawn(async move {
        let mut best: HashMap<ZenohId, u8> = HashMap::new();
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

            let preference = locator_preference(discovered.addr, &discovery.ethernet_ipv4());
            if best
                .get(&discovered.zid)
                .is_some_and(|current| *current <= preference)
            {
                log::debug!(
                    "skipping {} for {:?}: already have a better or equal locator",
                    discovered.addr,
                    discovered.zid
                );
                continue;
            }

            let Ok(locator) =
                Locator::new("tcp", discovered.addr.to_string(), "").inspect_err(|e| {
                    log::warn!("failed to parse locator from addr: {e}");
                })
            else {
                continue;
            };

            log::info!(
                "connecting to discovered peer {} at {}",
                discovered.zid,
                discovered.addr
            );
            runtime
                .connect_peer(&discovered.zid.into(), &[locator])
                .await;
            best.insert(discovered.zid, preference);
        }
    })));
    Ok(Session { z, _jh })
}

/// Open zenoh, retrying IPv4-only listen if dual-stack bind fails (typical on
/// Linux when `[::]` already owns IPv4-mapped).
pub async fn open_with_listen_fallback(
    identity: &str,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    match cfg(identity, listen_port) {
        Ok(cfg) => match open(cfg, namespace, listen_port, discovery_service_port).await {
            Ok(session) => Ok(session),
            Err(e) => {
                log::warn!(
                    "dual-stack zenoh listen failed ({e}); retrying IPv4 0.0.0.0:{listen_port}"
                );
                let cfg = cfg_with_listen(
                    identity,
                    listen_port,
                    &ipv4_only_listen_endpoints(listen_port),
                )?;
                open(cfg, namespace, listen_port, discovery_service_port).await
            }
        },
        Err(e) => Err(e),
    }
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
    _jh: Arc<AbortOnDrop>,
}
