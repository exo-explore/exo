use std::{
    env,
    ops::{Deref, DerefMut},
    panic,
};

use netwatcher::WatchHandle;
use tokio::{sync::mpsc, task::JoinHandle};
use zenoh::{Result, Session as ZSession, config::WhatAmI, internal::runtime::Runtime};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

pub mod swarm;

pub fn cfg(identity: u128, listen_port: u16) -> Result<zenoh::Config> {
    let namespace = env::var("EXO_ZENOH_NAMESPACE").unwrap_or_else(|_| "exo".to_string());
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity:x}\""))?;
    cfg.insert_json5("mode", "\"peer\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    cfg.insert_json5("scouting/multicast/enabled", "true")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    cfg.insert_json5("namespace", &format!("{namespace:?}"))?;
    cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
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

pub async fn open(cfg: zenoh::Config) -> Result<Session> {
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let session = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let _watch_all_handle = watch_all(runtime).await?;
    Ok(Session {
        session,
        _watch_all_handle,
    })
}
async fn watch_all(runtime: Runtime) -> Result<WatchAllHandle> {
    log::info!("spawning scout");
    let mut cfg = Config::default();
    cfg.insert_json5("scouting/multicast/ttl", "3")?;
    cfg.insert_json5("scouting/multicast/interface", "\"auto\"")?;
    let mut scout = zenoh::scout(WhatAmI::Peer, cfg.clone()).await?;
    let (send, mut recv) = mpsc::unbounded_channel();
    let _sync = netwatcher::watch_interfaces_with_callback(move |u| _ = send.send(u))?;
    let _async = tokio::task::spawn(async move {
        loop {
            tokio::select! {
                u = recv.recv() => {
                    if u.is_none() {
                        return Ok(());
                    }
                    log::info!("reloading scout");
                    scout = zenoh::scout(WhatAmI::Peer, cfg.clone()).await?;
                }
                hello = scout.recv_async() => {
                    if let Ok(hello) = hello {
                        // todo: auth
                        runtime
                            .connect_peer(&hello.zid().into(), hello.locators())
                            .await;
                    }
                }
            }
        }
    });
    Ok(WatchAllHandle { _sync, _async })
}

pub struct Session {
    pub session: ZSession,
    _watch_all_handle: WatchAllHandle,
}
impl Deref for Session {
    type Target = ZSession;
    fn deref(&self) -> &Self::Target {
        &self.session
    }
}
impl DerefMut for Session {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.session
    }
}
impl Drop for WatchAllHandle {
    fn drop(&mut self) {
        self._async.abort();
    }
}

pub struct WatchAllHandle {
    _sync: WatchHandle,
    _async: JoinHandle<Result<()>>,
}
