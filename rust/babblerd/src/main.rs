use std::io;

use babblerd::babel::handle_listener;
use tokio::{
    net::UnixListener,
    signal,
    sync::{broadcast, mpsc},
    task::{JoinHandle, JoinSet},
};

const PUBLIC_SOCK_PATH: &str = "/tmp/babblerd.sock";

enum State {
    Idle,
    Active {
        recv: broadcast::Receiver<String>,
        babel: JoinHandle<babblerd::Result<()>>,
        watcher: JoinHandle<babblerd::Result<()>>,
        listeners: JoinSet<io::Result<()>>,
    },
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    if !std::env::var("PATH").unwrap().contains("babeld") {
        tracing::error!("babeld not found on path");
        Err(babblerd::Error::Other(
            "babeld not found on path".to_string(),
        ))?;
    }
    let res = inner_main().await;
    _ = std::fs::remove_file(PUBLIC_SOCK_PATH);
    res
}

async fn inner_main() -> color_eyre::Result<()> {
    if let Err(e) = std::fs::remove_file(PUBLIC_SOCK_PATH)
        && e.kind() != io::ErrorKind::NotFound
    {
        return Err(e.into());
    }
    let public_socket = UnixListener::bind(PUBLIC_SOCK_PATH)?;
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
                        let (br_send, br_recv) = broadcast::channel(20);
                        let (mp_send, mp_recv) = mpsc::channel(20);
                        let babel = tokio::spawn(babblerd::babel::babel(mp_recv, br_send));
                        let watcher = tokio::spawn(babblerd::if_watcher::watch(mp_send));
                        let mut listeners = JoinSet::new();
                        listeners.spawn(handle_listener(sock, br_recv.resubscribe()));
                        State::Active { recv: br_recv, babel, watcher, listeners }
                    }
                }
            }
            State::Active {
                recv,
                mut babel,
                mut watcher,
                mut listeners,
            } => {
                tokio::select! {
                    sig = signal::ctrl_c() => {
                        sig?;
                        drop(recv);
                        watcher.abort();
                        babel.await??;
                        _ = watcher.await;
                        while let Some(res) = listeners.join_next().await {
                            res??;
                        }
                        break;
                    }
                    next_join_result = listeners.join_next(), if !listeners.is_empty() => {
                        next_join_result.expect("checked")??;
                        tracing::info!("dropped a listener");
                        if listeners.is_empty() {
                            tracing::info!("stopping babeld");
                            drop(recv);

                            watcher.abort();
                            _ = watcher.await;
                            babel.await??;

                            State::Idle
                        } else {
                            State::Active { recv, babel, watcher, listeners }
                        }
                    }
                    sock = public_socket.accept() => {
                        listeners.spawn(handle_listener(sock?.0, recv.resubscribe()));
                        State::Active { recv, babel, watcher, listeners }
                    }
                    res = &mut watcher => {
                        res??;
                        drop(recv);
                        babel.await??;
                        while let Some(res) = listeners.join_next().await {
                            res??;
                        }
                        State::Idle
                    }
                    res = &mut babel => {
                        res??;
                        drop(recv);
                        watcher.abort();
                        _ = watcher.await;
                        while let Some(res) = listeners.join_next().await {
                            res??;
                        }
                        State::Idle
                    }
                }
            }
        };
    }
    Ok(())
}
