use std::{fs::Permissions, io, os::unix::fs::PermissionsExt};

use babblerd::babel::handle_listener;
use color_eyre::eyre::WrapErr;
use tokio::{
    net::UnixListener,
    signal,
    sync::{broadcast, mpsc},
    task::{JoinHandle, JoinSet},
};

#[cfg(target_os = "macos")]
const PUBLIC_DIR: &str = "/var/run/babbler";
#[cfg(target_os = "linux")]
const PUBLIC_DIR: &str = "/run/babbler";
#[cfg(target_os = "macos")]
const PUBLIC_SOCK_PATH: &str = "/var/run/babbler/babblerd.sock";
#[cfg(target_os = "linux")]
const PUBLIC_SOCK_PATH: &str = "/run/babbler/babblerd.sock";

enum State {
    Idle,
    Active {
        recv: broadcast::Receiver<String>,
        babel: JoinHandle<babblerd::Result<()>>,
        watcher: JoinHandle<babblerd::Result<()>>,
        listeners: JoinSet<()>,
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
        return Err(babblerd::BabbleError::Other("babeld not found on path".to_owned()).into());
    }
    // cleanup old data
    match std::fs::remove_file(PUBLIC_SOCK_PATH) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => {
            return Err(e.into());
        }
        Ok(()) => {
            tracing::info!("cleaned up old file at {PUBLIC_SOCK_PATH}")
        }
        _ => {}
    }
    std::fs::create_dir_all(PUBLIC_DIR)?;
    // make our directory world readable
    std::fs::set_permissions(PUBLIC_DIR, Permissions::from_mode(0o0755))?;
    let res = inner_main().await;
    _ = std::fs::remove_file(PUBLIC_SOCK_PATH);
    res
}

async fn inner_main() -> color_eyre::Result<()> {
    tracing::info!("creating socket at {PUBLIC_SOCK_PATH}");
    let public_socket = UnixListener::bind(PUBLIC_SOCK_PATH)?;
    // make our socket world accessible
    std::fs::set_permissions(PUBLIC_SOCK_PATH, Permissions::from_mode(0o0666))?;
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
                        let (br_send, br_recv) = broadcast::channel(32);
                        let (mp_send, mp_recv) = mpsc::channel(32);
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
                        if let Ok(e) = watcher.await {
                            e.wrap_err("while ctrl-c")?
                        }
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
                                e.wrap_err("while closing listeners")?
                            }
                            babel.await?.wrap_err("while closing listeners")?;

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
                            e.wrap_err("while closing babeld")?
                        }
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
