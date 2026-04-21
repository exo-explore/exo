//! Major TODO: at some point don't call it "babbler" because that is a silly name that makes no sense
//!             but this is at the very bottom of my concerns right now :)

use std::{fs::Permissions, io, net::Ipv6Addr, os::unix::fs::PermissionsExt, sync::Arc};

use babblerd::tun::UtunDevice;
use babblerd::{
    babel::{handle_listener, BabelState},
    config::Config,
    if_watcher,
};
use color_eyre::eyre::{eyre, WrapErr};
use ipnet::Ipv6Net;
use n0_watcher::Watcher;
use netwatch::netmon;
use tokio::{
    net::UnixListener,
    signal,
    sync::{broadcast, mpsc, watch},
    task::{JoinHandle, JoinSet},
};

enum State {
    Idle,
    Active {
        recv: broadcast::Receiver<String>,
        utun: UtunDevice, // TODO: either rename it to TUN because its cross-platform OR if this is a macOS-only hack then figure out better cross-platform abstractions
        babel: JoinHandle<babblerd::Result<()>>,
        watcher: JoinHandle<babblerd::Result<()>>,
        state_logger: JoinHandle<()>,
        listeners: JoinSet<()>,
    },
}

async fn log_babel_state(mut recv: watch::Receiver<Arc<BabelState>>) {
    while recv.changed().await.is_ok() {
        let snapshot = recv.borrow_and_update();
        tracing::info!(state = ?*snapshot, "babel state snapshot updated");
    }
    tracing::info!("babel state stream closed");
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let config = Config::from_env()?;
    // cleanup old data
    match std::fs::remove_file(&config.public_socket_path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => {
            return Err(e.into());
        }
        Ok(()) => {
            tracing::info!(
                "cleaned up old file at {}",
                config.public_socket_path.display()
            );
        }
        _ => {}
    }
    std::fs::create_dir_all(&config.public_dir)?;
    // make our directory world readable
    if let Err(e) = std::fs::set_permissions(&config.public_dir, Permissions::from_mode(0o0755)) {
        if e.kind() == io::ErrorKind::PermissionDenied {
            return Err(eyre!(
                "Insufficient permissions to run daemon -- did you forget sudo?"
            ));
        }
        return Err(e.into());
    }

    let res = inner_main(&config).await;
    _ = std::fs::remove_file(&config.public_socket_path);
    res
}

async fn inner_main(config: &Config) -> color_eyre::Result<()> {
    tracing::info!("creating socket at {}", config.public_socket_path.display());
    tracing::info!(
        "router defaults: udp_port={} node_id_file={} app_prefix={}",
        config.router_udp_port,
        config.node_id_file.display(),
        config.exo_ula_prefix
    );
    let public_socket = UnixListener::bind(&config.public_socket_path)?;
    // make our socket world accessible
    std::fs::set_permissions(&config.public_socket_path, Permissions::from_mode(0o0666))?;
    let mut babbler: State = State::Idle;

    loop {
        babbler = match babbler {
            State::Idle => {
                tracing::info!("waiting for connection");
                tokio::select! {
                    sig = signal::ctrl_c() => {
                        sig?;
                        break;
                    }
                    sock = public_socket.accept() => {
                        let sock = sock?.0;
                        tracing::info!("starting babeld");
                        let (br_send, br_recv) = broadcast::channel(1024);
                        let (state_send, state_recv) =
                            watch::channel(Arc::new(BabelState::new()));
                        let (mp_send, mp_recv) = mpsc::channel(32);

                        // node id is a PREFIX/64, NODE_ID/48 and an INTERFACE_ID/16
                        // TODO: no longer we need interfaces :)
                        let ip_node_id = u128::from(rand::random::<u64>() & (u64::MAX>>16)) << 16;
                        let my_range = Ipv6Net::new_assert(
                            Ipv6Addr::from_bits(babblerd::PREFIX.addr().to_bits() | ip_node_id),
                            112,
                        );

                        // create address (node-ID) then launch tunnel device with that address,
                        // then advertise this tunnel device via babeld
                        let advertised = if_watcher::advertised_addr(my_range);
                        let utun = UtunDevice::create(advertised.addr())
                            .wrap_err("creating utun for advertised address")?;tracing::info!(
                            "created {} with advertised address {}",
                            utun.ifname(),
                            advertised
                        );
                        let babel = tokio::spawn(babblerd::babel(advertised, mp_recv, br_send, state_send));
                        let watcher = tokio::spawn(babblerd::watch(my_range, mp_send));
                        let state_logger = tokio::spawn(log_babel_state(state_recv));
                        let mut listeners = JoinSet::new();
                        listeners.spawn(handle_listener(sock, br_recv.resubscribe()));
                        State::Active { recv: br_recv, utun, babel, watcher, state_logger, listeners }
                    }
                }
            }
            State::Active {
                recv,
                utun,
                mut babel,
                mut watcher,
                state_logger,
                mut listeners,
            } => {
                tokio::select! {
                    sig = signal::ctrl_c() => {
                        sig?;
                        drop(recv);
                        watcher.abort();
                        if let Ok(e) = watcher.await {
                            e.wrap_err("while ctrl-c")?;
                        }
                        state_logger.abort();
                        let _ = state_logger.await;
                        babel.await?.wrap_err("while ctrl-c")?;
                        while let Some(res) = listeners.join_next().await {
                            res.wrap_err("while ctrl-c")?;
                        }
                        break;
                    }
                    next_join_result = listeners.join_next(), if !listeners.is_empty() => {
                        next_join_result.expect("checked")?;
                        tracing::info!("dropped a listener");
                        if listeners.is_empty() {
                            tracing::info!("stopping babeld");
                            drop(recv);

                            watcher.abort();
                            if let Ok(e) = watcher.await {
                                e.wrap_err("while closing listeners")?;
                            }
                            state_logger.abort();
                            let _ = state_logger.await;
                            babel.await?.wrap_err("while closing listeners")?;

                            State::Idle
                        } else {
                            State::Active { recv, utun, babel, watcher, state_logger, listeners }
                        }
                    }
                    sock = public_socket.accept() => {
                        listeners.spawn(handle_listener(sock?.0, recv.resubscribe()));
                        State::Active { recv, utun, babel, watcher, state_logger, listeners }
                    }
                    res = &mut watcher => {
                        res??;
                        drop(recv);
                        state_logger.abort();
                        let _ = state_logger.await;
                        babel.await?.wrap_err("while closing watcher")?;
                        while let Some(res2) = listeners.join_next().await {
                            res2.wrap_err("while closing watcher")?;
                        }
                        State::Idle
                    }
                    res = &mut babel => {
                        res??;
                        drop(recv);
                        watcher.abort();
                        if let Ok(e) = watcher.await {
                            e.wrap_err("while closing babeld")?;
                        }
                        state_logger.abort();
                        let _ = state_logger.await;
                        while let Some(res2) = listeners.join_next().await {
                            res2.wrap_err("while closing babeld")?;
                        }
                        State::Idle
                    }
                }
            }
        };
    }
    Ok(())
}
