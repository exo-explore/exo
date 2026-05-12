use std::env;

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;

pub mod discovery;
pub mod swarm;

pub fn cfg(identity: u128, listen_port: u16) -> Result<zenoh::Config> {
    assert!(listen_port != 0, "must used defined listen port port");
    let namespace = env::var("EXO_ZENOH_NAMESPACE").unwrap_or_else(|_| "exo".to_string());
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity:x}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    cfg.insert_json5("namespace", &format!("{namespace:?}"))?;
    cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    cfg.insert_json5("timestamping/enabled", "true")?;
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

pub async fn open(cfg: zenoh::Config, listen_port: u16) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let mut discovery = Discovery::new(z.zid(), listen_port).await?;
    let _jh = tokio::task::spawn(async move {
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
                    log::warn!("failed to pass locator from addr: {e}");
                })
            else {
                continue;
            };

            runtime
                .connect_peer(&discovered.zid.into(), &[locator])
                .await;
        }
    });
    Ok(Session { z, _jh })
}

pub struct Session {
    pub z: ZSession,
    _jh: JoinHandle<()>,
}
impl Drop for Session {
    fn drop(&mut self) {
        self._jh.abort();
    }
}
