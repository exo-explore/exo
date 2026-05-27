// Major TODO: at some point don't call it "babbler" because that is a silly name that makes no sense
//             but this is at the very bottom of my concerns right now :)

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("babblerd is mac/linux-only");

use std::{env, fs::Permissions, io, os::unix::fs::PermissionsExt, sync::Arc};

use babblerd::{
    babel::BabelState,
    config::{Config, TUN_MTU_ENV, TransportMode, default_tun_mtu},
    daemon, identity,
    tun::TunDevice,
};
use clap::Parser;
use color_eyre::eyre::{self, WrapErr, eyre};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
    net::UnixStream,
    signal,
    sync::watch,
    task::JoinSet,
    time::{Duration, sleep},
};

const INTERNAL_KEEPALIVE_TTL_MS: u64 = 30_000;
const INTERNAL_KEEPALIVE_INTERVAL_MS: u64 = 10_000;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, value_parser = parse_transport_mode)]
    router_transport: Option<TransportMode>,
    #[arg(long, conflicts_with = "router_transport")]
    force_tcp: bool,
    #[arg(long, value_parser = parse_tun_mtu)]
    tun_mtu: Option<u16>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let mut config = Config::from_env()?;
    let tun_mtu_from_env = env::var_os(TUN_MTU_ENV).is_some();
    let transport_overridden = cli.router_transport.is_some() || cli.force_tcp;
    if let Some(router_transport) = cli.router_transport {
        config.router_transport = router_transport;
    }
    if cli.force_tcp {
        config.router_transport = TransportMode::Tcp;
    }
    if transport_overridden && !tun_mtu_from_env && cli.tun_mtu.is_none() {
        config.tun_mtu = default_tun_mtu(config.router_transport);
    }
    if let Some(tun_mtu) = cli.tun_mtu {
        config.tun_mtu = tun_mtu;
    }

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
    let tun = TunDevice::create(node_addr.addr(), config.tun_mtu)
        .wrap_err("creating tun for node address")?;

    tracing::info!("creating socket at {}", config.public_socket_path.display());
    tracing::info!(
        "router defaults: transport={} port={} tun_mtu={} tcp_batch_target_bytes={} tcp_socket_buffer_bytes={} node_id_file={} app_prefix={} node_id={:#018x} node_addr={} tun={}",
        config.router_transport,
        config.router_udp_port,
        config.tun_mtu,
        config.tcp_batch_target_bytes,
        config.tcp_socket_buffer_bytes,
        config.node_id_file.display(),
        config.exo_ula_prefix,
        node_id,
        node_addr,
        tun.ifname(),
    );
    if config.router_transport == TransportMode::Tcp {
        tracing::warn!(
            tun_mtu = config.tun_mtu,
            tcp_batch_target_bytes = config.tcp_batch_target_bytes,
            tcp_socket_buffer_bytes = config.tcp_socket_buffer_bytes,
            "forced TCP transport requires matching TUN MTU on every peer; larger received frames are rejected"
        );
    }

    let public_socket = UnixListener::bind(&config.public_socket_path)?;

    // make our socket world accessible
    std::fs::set_permissions(&config.public_socket_path, Permissions::from_mode(0o0666))?;

    let (babel_state_send, _) = watch::channel(Arc::new(BabelState::new()));
    let (daemon, mut core_task) = daemon::DaemonCore::spawn(
        node_id,
        config.exo_ula_prefix,
        config.router_udp_port,
        config.router_transport,
        config.tun_mtu,
        config.tcp_batch_target_bytes,
        config.tcp_socket_buffer_bytes,
        node_addr,
        tun,
        babel_state_send,
    );
    // TEMP: keep the daemon alive without an external client until the real
    // frontend/test harness exists. This should be removed later.
    let mut internal_keepalive =
        tokio::spawn(internal_keepalive_client(config.public_socket_path.clone()));
    let mut listeners = JoinSet::new();

    loop {
        tokio::select! {
            sig = signal::ctrl_c() => {
                sig?;
                internal_keepalive.abort();
                let _ = (&mut internal_keepalive).await;
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
                listeners.spawn(daemon::handle_client(sock, daemon.clone()));
            }
            res = &mut core_task => {
                res??;
                internal_keepalive.abort();
                let _ = (&mut internal_keepalive).await;
                listeners.abort_all();
                while let Some(res2) = listeners.join_next().await {
                    res2.wrap_err("while closing daemon core")?;
                }
                break;
            }
            res = &mut internal_keepalive => {
                return Err(eyre!("internal keepalive client exited unexpectedly: {res:?}"));
            }
            next_join_result = listeners.join_next(), if !listeners.is_empty() => {
                next_join_result.expect("checked")?;
                tracing::info!("dropped a listener");
            }
        }
    }

    Ok(())
}

fn parse_transport_mode(raw: &str) -> Result<TransportMode, String> {
    raw.parse()
}

fn parse_tun_mtu(raw: &str) -> Result<u16, String> {
    babblerd::config::parse_tun_mtu(raw)
}

async fn internal_keepalive_client(socket_path: std::path::PathBuf) {
    loop {
        match UnixStream::connect(&socket_path).await {
            Ok(stream) => {
                tracing::info!(
                    socket=%socket_path.display(),
                    "internal keepalive client connected"
                );
                let (reader, mut writer) = stream.into_split();
                let mut reader = BufReader::new(reader).lines();

                match reader.next_line().await {
                    Ok(Some(line)) => {
                        tracing::debug!(?line, "internal keepalive initial state");
                    }
                    Ok(None) => {
                        tracing::warn!("internal keepalive connection closed before initial state");
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                    Err(err) => {
                        tracing::warn!(error=%err, "internal keepalive failed to read initial state");
                        sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                }

                loop {
                    let command = format!("keepalive {INTERNAL_KEEPALIVE_TTL_MS}\n");
                    if let Err(err) = writer.write_all(command.as_bytes()).await {
                        tracing::warn!(error=%err, "internal keepalive failed to send keepalive");
                        break;
                    }

                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            tracing::debug!(?line, "internal keepalive response");
                        }
                        Ok(None) => {
                            tracing::warn!("internal keepalive connection closed");
                            break;
                        }
                        Err(err) => {
                            tracing::warn!(error=%err, "internal keepalive failed to read response");
                            break;
                        }
                    }

                    sleep(Duration::from_millis(INTERNAL_KEEPALIVE_INTERVAL_MS)).await;
                }
            }
            Err(err) => {
                tracing::warn!(error=%err, socket=%socket_path.display(), "internal keepalive failed to connect");
            }
        }

        sleep(Duration::from_secs(1)).await;
    }
}
