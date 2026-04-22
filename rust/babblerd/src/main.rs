// Major TODO: at some point don't call it "babbler" because that is a silly name that makes no sense
//             but this is at the very bottom of my concerns right now :)

#[cfg(not(unix))]
compile_error!("babblerd is unix-only");

use std::{fs::Permissions, io, os::unix::fs::PermissionsExt, sync::Arc};

use babblerd::{babel::BabelState, config::Config, daemon, identity, tun::UtunDevice};
use color_eyre::eyre::{self, WrapErr, eyre};
use tokio::{
    net::UnixListener,
    signal,
    sync::{broadcast, watch},
    task::JoinSet,
};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let config = Config::from_env()?;

    // cleanup old  public socket path
    match std::fs::remove_file(&config.public_socket_path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
        Ok(()) => {
            tracing::info!(
                "cleaned up old file at {}",
                config.public_socket_path.display()
            );
        }
        _ => {}
    }

    // create new public directory
    std::fs::create_dir_all(&config.public_dir)?;
    if let Err(e) = std::fs::set_permissions(&config.public_dir, Permissions::from_mode(0o0755)) {
        if e.kind() == io::ErrorKind::PermissionDenied {
            return Err(eyre!(
                "Insufficient permissions to run daemon -- did you forget sudo?"
            ));
        }
        return Err(e.into());
    }

    let res = inner_main(&config).await;
    let _ = std::fs::remove_file(&config.public_socket_path);
    res
}

async fn inner_main(config: &Config) -> eyre::Result<()> {
    let node_id = identity::load_or_create_node_id(&config.node_id_file)?;
    let node_addr = identity::node_addr(config.exo_ula_prefix, node_id)?;
    let utun = UtunDevice::create(node_addr.addr()).wrap_err("creating utun for node address")?;

    tracing::info!("creating socket at {}", config.public_socket_path.display());
    tracing::info!(
        "router defaults: udp_port={} node_id_file={} app_prefix={} node_id={:#018x} node_addr={} utun={}",
        config.router_udp_port,
        config.node_id_file.display(),
        config.exo_ula_prefix,
        node_id,
        node_addr,
        utun.ifname(),
    );

    let public_socket = UnixListener::bind(&config.public_socket_path)?;

    // make our socket world accessible
    std::fs::set_permissions(&config.public_socket_path, Permissions::from_mode(0o0666))?;

    let (line_send, _) = broadcast::channel(1024);
    let (babel_state_send, _) = watch::channel(Arc::new(BabelState::new()));
    let (daemon, mut core_task) = daemon::DaemonCore::spawn(
        node_id,
        node_addr,
        utun,
        line_send.clone(),
        babel_state_send,
    );
    let mut listeners = JoinSet::new();

    loop {
        tokio::select! {
            sig = signal::ctrl_c() => {
                sig?;
                listeners.abort_all();
                while let Some(res) = listeners.join_next().await {
                    res.wrap_err("while ctrl-c")?;
                }
                drop(daemon);
                core_task.await??;
                break;
            }
            sock = public_socket.accept() => {
                let sock = sock?.0;
                listeners.spawn(daemon::handle_client(sock, daemon.clone(), line_send.subscribe()));
            }
            res = &mut core_task => {
                res??;
                listeners.abort_all();
                while let Some(res2) = listeners.join_next().await {
                    res2.wrap_err("while closing daemon core")?;
                }
                break;
            }
            next_join_result = listeners.join_next(), if !listeners.is_empty() => {
                next_join_result.expect("checked")?;
                tracing::info!("dropped a listener");
            }
        }
    }

    Ok(())
}
