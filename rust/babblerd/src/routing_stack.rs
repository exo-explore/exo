use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use ipnet::Ipv6Net;
use tokio::{
    sync::{broadcast, mpsc, watch},
    task::JoinHandle,
};

use crate::babel::BabelState;
use crate::daemon::{RoutingStackEvent, StackTaskKind};

pub struct RoutingStack {
    babel: JoinHandle<crate::Result<()>>,
    watcher: JoinHandle<crate::Result<()>>,
    state_logger: JoinHandle<()>,
}

impl RoutingStack {
    pub async fn start(
        node_addr: Ipv6Net,
        line_send: broadcast::Sender<String>,
        state_send: watch::Sender<Arc<BabelState>>,
        event_send: mpsc::Sender<RoutingStackEvent>,
    ) -> Result<Self> {
        let (iface_send, iface_recv) = mpsc::channel(32);
        let mut state_recv = state_send.subscribe();

        let state_logger = tokio::spawn(async move {
            while state_recv.changed().await.is_ok() {
                let snapshot = state_recv.borrow_and_update();
                tracing::info!(state = ?*snapshot, "babel state snapshot updated");
            }
            tracing::info!("babel state stream closed");
        });

        let babel_events = event_send.clone();
        let babel = tokio::spawn(async move {
            let res = crate::babel(node_addr, iface_recv, line_send, state_send).await;
            let _ = babel_events
                .send(RoutingStackEvent::Exited {
                    kind: StackTaskKind::Babel,
                    error: res.as_ref().err().map(ToString::to_string),
                })
                .await;
            res
        });

        let watcher_events = event_send;
        let watcher = tokio::spawn(async move {
            let res = crate::watch(iface_send).await;
            let _ = watcher_events
                .send(RoutingStackEvent::Exited {
                    kind: StackTaskKind::Watcher,
                    error: res.as_ref().err().map(ToString::to_string),
                })
                .await;
            res
        });

        Ok(Self {
            babel,
            watcher,
            state_logger,
        })
    }

    pub async fn stop(self) -> Result<()> {
        let Self {
            babel,
            watcher,
            state_logger,
        } = self;

        watcher.abort();
        if let Ok(res) = watcher.await {
            res.wrap_err("stopping interface watcher")?;
        }

        state_logger.abort();
        let _ = state_logger.await;

        babel.await?.wrap_err("stopping babeld runtime")?;
        Ok(())
    }
}
