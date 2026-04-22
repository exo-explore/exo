use color_eyre::eyre::{Result, WrapErr, eyre};
use ipnet::Ipv6Net;
use std::fmt::{Display, Formatter};
use std::{future::pending, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    sync::{broadcast, mpsc, oneshot, watch},
    task::JoinHandle,
    time::{Duration, Instant},
};

use crate::routing_stack::RoutingStack;
use crate::{babel::BabelState, tun::UtunDevice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Off,
    Starting,
    On,
    Stopping,
}

impl Display for ServiceState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Starting => write!(f, "starting"),
            Self::On => write!(f, "on"),
            Self::Stopping => write!(f, "stopping"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DaemonStatus {
    pub service_state: ServiceState,
    pub node_id: u64,
    pub node_addr: Ipv6Net,
    pub utun_ifname: Arc<str>,
    // realistically should always have one?? right??
    pub keepalive_deadline: Option<Instant>,
    pub last_error: Option<Arc<str>>,
}

impl DaemonStatus {
    fn new(node_id: u64, node_addr: Ipv6Net, utun_ifname: Arc<str>) -> Self {
        Self {
            service_state: ServiceState::Off,
            node_id,
            node_addr,
            utun_ifname,
            keepalive_deadline: None,
            last_error: None,
        }
    }

    pub fn render(&self) -> String {
        let keepalive_remaining_ms = self
            .keepalive_deadline
            .and_then(|deadline| deadline.checked_duration_since(Instant::now()))
            .map(|remaining| remaining.as_millis().to_string())
            .unwrap_or_else(|| "none".to_owned());

        let mut line = format!(
            "state {} node_id={:#018x} node_addr={} utun={} keepalive_remaining_ms={}",
            self.service_state,
            self.node_id,
            self.node_addr,
            self.utun_ifname,
            keepalive_remaining_ms
        );
        if let Some(err) = &self.last_error {
            line.push_str(&format!(" last_error={err:?}"));
        }
        line
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackTaskKind {
    Babel,
    Watcher,
}

#[derive(Debug)]
pub enum RoutingStackEvent {
    Exited {
        kind: StackTaskKind,
        error: Option<String>,
    },
}

#[derive(Debug)]
enum DaemonCommand {
    KeepAlive {
        ttl: Duration,
        reply: oneshot::Sender<Result<DaemonStatus>>,
    },
    GetState {
        reply: oneshot::Sender<DaemonStatus>,
    },
}

#[derive(Clone)]
pub struct DaemonHandle {
    send: mpsc::Sender<DaemonCommand>,
}

impl DaemonHandle {
    pub async fn keep_alive(&self, ttl: Duration) -> Result<DaemonStatus> {
        let (reply_send, reply_recv) = oneshot::channel();
        self.send
            .send(DaemonCommand::KeepAlive {
                ttl,
                reply: reply_send,
            })
            .await
            .map_err(|_| eyre!("daemon core stopped"))?;
        reply_recv.await.map_err(|_| eyre!("daemon core stopped"))?
    }

    pub async fn get_state(&self) -> Result<DaemonStatus> {
        let (reply_send, reply_recv) = oneshot::channel();
        self.send
            .send(DaemonCommand::GetState { reply: reply_send })
            .await
            .map_err(|_| eyre!("daemon core stopped"))?;
        reply_recv.await.map_err(|_| eyre!("daemon core stopped"))
    }
}

pub struct DaemonCore {
    status: DaemonStatus,
    _utun: UtunDevice,
    routing_stack: Option<RoutingStack>,
    line_send: broadcast::Sender<String>,
    babel_state_send: watch::Sender<Arc<BabelState>>,
    command_recv: mpsc::Receiver<DaemonCommand>,
    event_send: mpsc::Sender<RoutingStackEvent>,
    event_recv: mpsc::Receiver<RoutingStackEvent>,
}

impl DaemonCore {
    pub fn spawn(
        node_id: u64,
        node_addr: Ipv6Net,
        utun: UtunDevice,
        line_send: broadcast::Sender<String>,
        babel_state_send: watch::Sender<Arc<BabelState>>,
    ) -> (DaemonHandle, JoinHandle<Result<()>>) {
        let (command_send, command_recv) = mpsc::channel(32);
        let (event_send, event_recv) = mpsc::channel(8);
        let status = DaemonStatus::new(node_id, node_addr, Arc::from(utun.ifname().to_owned()));

        let core = Self {
            status,
            _utun: utun,
            routing_stack: None,
            line_send,
            babel_state_send,
            command_recv,
            event_send,
            event_recv,
        };

        let handle = DaemonHandle { send: command_send };
        let task = tokio::spawn(core.run());
        (handle, task)
    }

    async fn run(mut self) -> Result<()> {
        loop {
            tokio::select! {
                command = self.command_recv.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.handle_command(command).await?;
                }
                event = self.event_recv.recv() => {
                    let Some(event) = event else {
                        break;
                    };
                    self.handle_stack_event(event).await?;
                }
                _ = lease_timer(self.status.keepalive_deadline) => {
                    self.handle_lease_expiry().await?;
                }
            }
        }

        self.stop_stack().await?;
        Ok(())
    }

    async fn handle_command(&mut self, command: DaemonCommand) -> Result<()> {
        match command {
            DaemonCommand::KeepAlive { ttl, reply } => {
                self.status.keepalive_deadline = Some(Instant::now() + ttl);
                if self.routing_stack.is_none() {
                    let result = self.start_stack().await.map(|()| self.status.clone());
                    let _ = reply.send(result);
                    return Ok(());
                }
                let _ = reply.send(Ok(self.status.clone()));
                Ok(())
            }
            DaemonCommand::GetState { reply } => {
                let _ = reply.send(self.status.clone());
                Ok(())
            }
        }
    }

    async fn handle_stack_event(&mut self, event: RoutingStackEvent) -> Result<()> {
        let RoutingStackEvent::Exited { kind, error } = event;
        if self.routing_stack.is_none() {
            return Ok(());
        }

        tracing::warn!(?kind, ?error, "routing stack task exited");
        self.status.last_error =
            Some(Arc::from(error.unwrap_or_else(|| {
                format!("{kind:?} task exited unexpectedly")
            })));
        if let Err(err) = self.stop_stack().await {
            self.status.last_error = Some(Arc::from(err.to_string()));
        }
        Ok(())
    }

    async fn handle_lease_expiry(&mut self) -> Result<()> {
        let expired = self
            .status
            .keepalive_deadline
            .is_some_and(|deadline| deadline <= Instant::now());
        if expired {
            tracing::info!("keepalive expired, transitioning routing stack off");
            self.status.keepalive_deadline = None;
            if let Err(err) = self.stop_stack().await {
                self.status.last_error = Some(Arc::from(err.to_string()));
            }
        }
        Ok(())
    }

    async fn start_stack(&mut self) -> Result<()> {
        if self.routing_stack.is_some() {
            return Ok(());
        }

        self.status.service_state = ServiceState::Starting;
        self.status.last_error = None;
        match RoutingStack::start(
            self.status.node_addr,
            self.line_send.clone(),
            self.babel_state_send.clone(),
            self.event_send.clone(),
        )
        .await
        {
            Ok(stack) => {
                self.routing_stack = Some(stack);
                self.status.service_state = ServiceState::On;
                Ok(())
            }
            Err(err) => {
                self.status.service_state = ServiceState::Off;
                self.status.last_error = Some(Arc::from(err.to_string()));
                Err(err)
            }
        }
    }

    async fn stop_stack(&mut self) -> Result<()> {
        let Some(stack) = self.routing_stack.take() else {
            self.status.service_state = ServiceState::Off;
            self.babel_state_send
                .send_replace(Arc::new(BabelState::new()));
            return Ok(());
        };

        self.status.service_state = ServiceState::Stopping;
        let stop_result = stack.stop().await;
        self.babel_state_send
            .send_replace(Arc::new(BabelState::new()));
        self.status.service_state = ServiceState::Off;
        if let Err(err) = stop_result {
            self.status.last_error = Some(Arc::from(err.to_string()));
            return Err(err);
        }
        Ok(())
    }
}

async fn lease_timer(deadline: Option<Instant>) {
    if let Some(deadline) = deadline {
        tokio::time::sleep_until(deadline).await;
    } else {
        pending::<()>().await;
    }
}

pub async fn handle_client(
    sock: UnixStream,
    daemon: DaemonHandle,
    mut lines: broadcast::Receiver<String>,
) {
    tracing::info!("new socket conn");
    let (reader, mut write) = sock.into_split();
    let mut reader = BufReader::new(reader).lines();

    if let Ok(state) = daemon.get_state().await {
        let _ = write
            .write_all(format!("{}\n", state.render()).as_bytes())
            .await;
    }

    loop {
        tokio::select! {
            read = reader.next_line() => {
                let Ok(Some(line)) = read else { break; };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let response = match handle_command_line(trimmed, &daemon).await {
                    Ok(response) => response,
                    Err(err) => format!("error {err}"),
                };

                if let Err(err) = write.write_all(format!("{response}\n").as_bytes()).await {
                    tracing::warn!(error=%err, "failed to write command response");
                    break;
                }
            }
            recv = lines.recv() => {
                let res = match recv {
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        tracing::warn!("receiver lagged");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::debug!("receiver closed, dropping connection");
                        break;
                    }
                    Ok(line) => write.write_all(format!("{line}\n").as_bytes()).await,
                };
                if let Err(err) = res {
                    tracing::warn!(error=%err, "failed to write babel line to client");
                    break;
                }
            }
        }
    }

    tracing::info!("closing socket conn");
    let _ = write.shutdown().await;
}

async fn handle_command_line(line: &str, daemon: &DaemonHandle) -> Result<String> {
    let mut parts = line.split_whitespace();
    let Some(command) = parts.next() else {
        return Ok("error empty-command".to_owned());
    };

    match command {
        "get-state" => Ok(daemon.get_state().await?.render()),
        "keepalive" => {
            let Some(ttl_ms) = parts.next() else {
                return Err(eyre!("keepalive requires ttl_ms"));
            };
            let ttl_ms = ttl_ms
                .parse::<u64>()
                .wrap_err_with(|| format!("invalid ttl_ms: {ttl_ms:?}"))?;
            Ok(daemon
                .keep_alive(Duration::from_millis(ttl_ms))
                .await?
                .render())
        }
        "help" => Ok("commands: get-state | keepalive <ttl_ms>".to_owned()),
        other => Err(eyre!("unknown command: {other}")),
    }
}
