use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use ipnet::Ipv6Net;
use tokio::{
    sync::{mpsc, watch},
    task::JoinHandle,
    time::Duration,
};

use crate::babel::BabelState;
use crate::config::TUN_MTU;
use crate::daemon::{RoutingStackEvent, StackTaskKind};
use crate::dataplane::{Dataplane, DataplaneConfig, DataplanePublisher, PublishSnapshotError};
use crate::fib::FibBuilder;
use crate::tun::TunDevice;

pub struct RoutingStack {
    babel: JoinHandle<crate::Result<()>>,
    watcher: JoinHandle<crate::Result<()>>,
    state_logger: JoinHandle<()>,
    fib_publisher: JoinHandle<crate::Result<()>>,
    dataplane_monitor: JoinHandle<()>,
    dataplane: Dataplane,
}

impl RoutingStack {
    pub async fn start(
        node_addr: Ipv6Net,
        tun: &TunDevice,
        udp_port: u16,
        state_send: watch::Sender<Arc<BabelState>>,
        event_send: mpsc::Sender<RoutingStackEvent>,
    ) -> Result<Self> {
        let (iface_send, iface_recv) = mpsc::channel(32);
        let mut state_recv = state_send.subscribe();
        let fib_state_recv = state_send.subscribe();
        let initial_state = state_send.borrow().clone();
        let mut dataplane = Dataplane::spawn(DataplaneConfig {
            tun_device: tun
                .try_clone_device()
                .wrap_err("cloning TUN device for dataplane")?,
            udp_port,
            initial_fib: Arc::new(
                FibBuilder::new([node_addr.addr()], TUN_MTU).derive(initial_state.as_ref()),
            ),
        })?;
        let dataplane_exit = dataplane
            .take_exit_receiver()
            .expect("dataplane exit receiver must exist immediately after spawn");

        let state_logger = tokio::spawn(async move {
            while state_recv.changed().await.is_ok() {
                let snapshot = state_recv.borrow_and_update();
                tracing::info!(state = ?*snapshot, "babel state snapshot updated");
            }
            tracing::info!("babel state stream closed");
        });

        let babel_events = event_send.clone();
        let babel_state_send = state_send.clone();
        let babel = tokio::spawn(async move {
            let res = crate::babel(node_addr, iface_recv, babel_state_send).await;
            let _ = babel_events
                .send(RoutingStackEvent::Exited {
                    kind: StackTaskKind::Babel,
                    error: res.as_ref().err().map(ToString::to_string),
                })
                .await;
            res
        });

        let watcher_events = event_send.clone();
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

        let fib_events = event_send.clone();
        let dataplane_publisher = dataplane.publisher();
        let fib_publisher = tokio::spawn(async move {
            let res = publish_fib_updates(node_addr, fib_state_recv, dataplane_publisher).await;
            let _ = fib_events
                .send(RoutingStackEvent::Exited {
                    kind: StackTaskKind::FibPublisher,
                    error: res.as_ref().err().map(ToString::to_string),
                })
                .await;
            res
        });

        let dataplane_events = event_send.clone();
        let dataplane_monitor = tokio::spawn(async move {
            let exit = dataplane_exit.await;
            let (kind, error) = match exit {
                Ok(Ok(())) => (StackTaskKind::Dataplane, None),
                Ok(Err(err)) => (StackTaskKind::Dataplane, Some(err)),
                Err(err) => (
                    StackTaskKind::Dataplane,
                    Some(format!("dataplane exit receiver dropped: {err}")),
                ),
            };
            let _ = dataplane_events
                .send(RoutingStackEvent::Exited { kind, error })
                .await;
        });

        Ok(Self {
            babel,
            watcher,
            state_logger,
            fib_publisher,
            dataplane_monitor,
            dataplane,
        })
    }

    pub async fn stop(self) -> Result<()> {
        let Self {
            babel,
            watcher,
            state_logger,
            fib_publisher,
            dataplane_monitor,
            dataplane,
        } = self;

        watcher.abort();
        if let Ok(res) = watcher.await {
            res.wrap_err("stopping interface watcher")?;
        }

        state_logger.abort();
        let _ = state_logger.await;

        fib_publisher.abort();
        let _ = fib_publisher.await;

        dataplane_monitor.abort();
        let _ = dataplane_monitor.await;

        dataplane.stop().wrap_err("stopping dataplane thread")?;

        babel.await?.wrap_err("stopping babeld runtime")?;
        Ok(())
    }
}

async fn publish_fib_updates(
    node_addr: Ipv6Net,
    mut state_recv: watch::Receiver<Arc<BabelState>>,
    publisher: DataplanePublisher,
) -> crate::Result<()> {
    let builder = FibBuilder::new([node_addr.addr()], TUN_MTU);
    let mut pending = Some(Arc::new(builder.derive(state_recv.borrow().as_ref())));

    loop {
        if let Some(snapshot) = pending.take() {
            match publisher.try_publish(snapshot) {
                Ok(()) => {}
                Err(PublishSnapshotError::Full(snapshot)) => {
                    pending = Some(snapshot);
                }
                Err(PublishSnapshotError::Stopped) => {
                    return Err(crate::BabbleError::Other(
                        "dataplane thread stopped".to_owned(),
                    ));
                }
            }
        }

        tokio::select! {
            changed = state_recv.changed() => {
                match changed {
                    Ok(()) => {
                        let snapshot = {
                            let state = state_recv.borrow_and_update();
                            Arc::new(builder.derive(state.as_ref()))
                        };
                        pending = Some(snapshot);
                    }
                    Err(_) => return Ok(()),
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(10)), if pending.is_some() => {}
        }
    }
}
