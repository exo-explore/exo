//! Managed `babeld` runtime for `babblerd`.
//!
//! This module owns the full lifecycle of the private `babeld` instance:
//!
//! - spawn-time configuration of the child process
//! - the private Unix socket path used for the local control connection
//! - connecting to that socket and speaking the local Babel protocol
//! - running the monitor-driven control loop
//! - shutdown and cleanup of the child process and socket
//!
//! Unlike the old `process` / `session` split, this is intended to model the real runtime unit:
//! a single managed `babeld` process together with its single local control session.

use std::fs::Permissions;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

use futures_lite::FutureExt;
use ipnet::Ipv6Net;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::time::Duration;

use crate::babel::Babble;
use crate::babel::command::BabelCommand;
use crate::babel::line::parse::ParseError;
use crate::babel::line::{self, BabelLine, HeaderLine, Status};
use crate::babel::state::BabelState;
use crate::{BabbleError, Result};

#[cfg(target_os = "macos")]
const PRIVATE_SOCK_PATH: &str = "/var/run/babbler/private/babeld.sock";
#[cfg(target_os = "linux")]
const PRIVATE_SOCK_PATH: &str = "/run/babbler/private/babeld.sock";
#[cfg(target_os = "macos")]
const PRIVATE_DIR: &str = "/var/run/babbler/private";
#[cfg(target_os = "linux")]
const PRIVATE_DIR: &str = "/run/babbler/private";

pub(crate) struct BabelRuntime {
    proc: Child,
    read: Lines<BufReader<OwnedReadHalf>>,
    write: OwnedWriteHalf,
    line_send: broadcast::Sender<String>,
    state_send: watch::Sender<Arc<BabelState>>,
    state: BabelState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupStage {
    Banner,
    Version,
    Host,
    MyId,
    Ready,
}

impl StartupStage {
    fn advance(self, line: BabelLine) -> Result<Option<Self>> {
        match (self, line) {
            (Self::Banner, BabelLine::Header(HeaderLine::Banner { major: 1, minor: 0 })) => {
                Ok(Some(Self::Version))
            }
            (Self::Version, BabelLine::Header(HeaderLine::Version(_))) => Ok(Some(Self::Host)),
            (Self::Host, BabelLine::Header(HeaderLine::Host(_))) => Ok(Some(Self::MyId)),
            (Self::MyId, BabelLine::Header(HeaderLine::MyId(_))) => Ok(Some(Self::Ready)),
            (Self::Ready, BabelLine::Status(Status::Ok)) => Ok(None),
            (stage, other) => Err(BabbleError::Other(format!(
                "unexpected babeld startup line while waiting for {stage:?}: {other:?}"
            ))),
        }
    }
}

impl Drop for BabelRuntime {
    #[inline]
    fn drop(&mut self) {
        // Emergency SIGKILL to avoid leaking an unmanaged babeld subprocess.
        match self.proc.try_wait() {
            Ok(None) => {}
            Ok(Some(sc)) => {
                if !sc.success() {
                    _ = self.proc.start_kill();
                }
            }
            _ => {
                _ = self.proc.start_kill();
            }
        }
    }
}

impl BabelRuntime {
    #[tracing::instrument(skip(line_send, state_send))]
    pub(crate) async fn spawn(
        advertised: Ipv6Net,
        iface: &str,
        line_send: broadcast::Sender<String>,
        state_send: watch::Sender<Arc<BabelState>>,
    ) -> Result<Self> {
        tokio::fs::create_dir_all(PRIVATE_DIR).await?;
        // TODO: remove this magic constant (and magic constants in general)
        tokio::fs::set_permissions(PRIVATE_DIR, Permissions::from_mode(0o0700)).await?;
        tracing::info!("spawning babeld socket in {PRIVATE_SOCK_PATH}");

        let mut proc = match Command::new("babeld")
            .arg("-G")
            .arg(PRIVATE_SOCK_PATH)
            .arg("-I")
            .arg(format!("{PRIVATE_DIR}/babeld.pid"))
            .arg("-C")
            .arg("kernel-install false")
            .arg("-C")
            .arg(format!("redistribute local ip {advertised}"))
            .arg("-C")
            .arg("redistribute local deny")
            .arg(iface)
            .spawn()
        {
            Ok(proc) => proc,
            Err(e) => {
                tracing::warn!(error=%e, "failed to spawn babeld");
                return Err(e.into());
            }
        };

        if let Err(err) = Self::wait_for_socket(&mut proc).await {
            Self::abort_child(&mut proc).await;
            return Err(err);
        }

        // TODO: magic undocumented number
        if let Err(err) =
            std::fs::set_permissions(PRIVATE_SOCK_PATH, Permissions::from_mode(0o0600))
        {
            Self::abort_child(&mut proc).await;
            return Err(err.into());
        }

        let (reader, write) = match UnixStream::connect(PRIVATE_SOCK_PATH).await {
            Ok(stream) => stream.into_split(),
            Err(err) => {
                Self::abort_child(&mut proc).await;
                return Err(err.into());
            }
        };

        let mut runtime = Self {
            proc,
            read: BufReader::new(reader).lines(),
            write,
            line_send,
            state_send,
            state: BabelState::new(),
        };

        if let Err(err) = runtime.await_ready().await {
            let _ = runtime.shutdown().await;
            return Err(err);
        }

        Ok(runtime)
    }

    async fn wait_for_socket(proc: &mut Child) -> Result<()> {
        // maybe spinning logic is fine with magic numbers...?
        tokio::time::sleep(Duration::from_millis(10)).await;
        while !matches!(tokio::fs::try_exists(PRIVATE_SOCK_PATH).await, Ok(true)) {
            if let Some(status) = proc.try_wait()? {
                return Err(BabbleError::BabeldCrashed(status.code()));
            }
            tracing::info!("where is the sock");
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Ok(())
    }

    async fn abort_child(proc: &mut Child) {
        let _ = proc.kill().await;
    }

    #[tracing::instrument(skip_all)]
    async fn await_ready(&mut self) -> Result<()> {
        let mut stage = StartupStage::Banner;
        while let Some(line) = self.read.next_line().await? {
            match self.observe_line(line)? {
                Ok(parsed) => match stage.advance(parsed)? {
                    Some(next) => stage = next,
                    None => {
                        tracing::info!("babeld ok");
                        return Ok(());
                    }
                },
                Err(err) => {
                    return Err(BabbleError::Other(format!(
                        "failed to parse babeld startup prelude: {err}"
                    )));
                }
            }
        }
        Err(BabbleError::Other(
            "babeld closed before completing startup prelude".into(),
        ))
    }

    #[tracing::instrument(skip(self))]
    async fn query(&mut self, cmd: &BabelCommand) -> io::Result<Option<Status>> {
        self.write.write_all(cmd.encode().as_bytes()).await?;
        loop {
            let Some(line) = self.read.next_line().await? else {
                tracing::warn!("babeld closed unexpectedly");
                return Ok(None);
            };
            match self.observe_line(line)? {
                Ok(parsed) => {
                    let status = self.reduce_live_line(parsed)?;
                    let Some(status) = status else {
                        continue;
                    };
                    match &status {
                        Status::Ok => {}
                        Status::Bad => tracing::warn!("malformed message sent to babeld"),
                        Status::No(rest) => tracing::warn!("message rejected: {rest:?}"),
                    }
                    return Ok(Some(status));
                }
                Err(err) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("failed to parse babeld command output: {err}"),
                    ));
                }
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn observe_line(&self, line: String) -> io::Result<std::result::Result<BabelLine, ParseError>> {
        tracing::info!("[babel] {:?}", line);

        let observed = match line::parse::parse_line(&line) {
            Ok(parsed) => {
                tracing::info!("[parsed] {:?}", parsed);
                Ok(parsed)
            }
            Err(err) => {
                tracing::error!(error=%err, "failed to parse babeld line");
                Err(err)
            }
        };

        self.line_send
            .send(line)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "babel listeners dropped"))?;
        Ok(observed)
    }

    #[tracing::instrument(skip(self))]
    async fn start_monitoring(&mut self) -> io::Result<Option<Status>> {
        let mut snapshot = BabelState::new();
        self.write
            .write_all(BabelCommand::Monitor.encode().as_bytes())
            .await?;
        loop {
            let Some(line) = self.read.next_line().await? else {
                tracing::warn!("babeld closed unexpectedly");
                return Ok(None);
            };
            match self.observe_line(line)? {
                Ok(BabelLine::Event(event)) => {
                    snapshot.apply(event);
                }
                Ok(BabelLine::Status(status)) => {
                    match &status {
                        Status::Ok => {
                            self.state = snapshot;
                            self.publish_state();
                        }
                        Status::Bad => tracing::warn!("malformed message sent to babeld"),
                        Status::No(rest) => tracing::warn!("message rejected: {rest:?}"),
                    }
                    return Ok(Some(status));
                }
                Ok(BabelLine::Header(header)) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected header line during monitor bootstrap: {header:?}"),
                    ));
                }
                Err(err) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("failed to parse babeld monitor bootstrap output: {err}"),
                    ));
                }
            }
        }
    }

    fn publish_state(&self) {
        self.state_send.send_replace(Arc::new(self.state.clone()));
    }

    fn reduce_live_line(&mut self, line: BabelLine) -> io::Result<Option<Status>> {
        match line {
            BabelLine::Event(event) => {
                self.state.apply(event);
                self.publish_state();
                Ok(None)
            }
            BabelLine::Status(status) => Ok(Some(status)),
            BabelLine::Header(header) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unexpected header line after startup: {header:?}"),
            )),
        }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn run(&mut self, mut recv: mpsc::Receiver<Babble>) -> Result<()> {
        match self.start_monitoring().await? {
            Some(Status::Ok) => {}
            Some(Status::Bad) => {
                return Err(BabbleError::Other(
                    "babeld rejected monitor command as malformed".into(),
                ));
            }
            Some(Status::No(reason)) => {
                return Err(BabbleError::Other(format!(
                    "babeld rejected monitor command: {reason:?}"
                )));
            }
            None => return Ok(()),
        }

        loop {
            tokio::select! {
                babble = recv.recv() => {
                    tracing::debug!("[babble] {:?}", babble);
                    let Some(babble) = babble else {
                        break;
                    };
                    match babble {
                        Babble::AddIface(iface) => {
                            let cmd = BabelCommand::Interface(iface);
                            self.query(&cmd).await?;
                        }
                    }
                },
                line = self.read.next_line() => {
                    let Ok(Some(line)) = line else {
                        break;
                    };
                    match self.observe_line(line)? {
                        Ok(parsed) => {
                            if let Some(status) = self.reduce_live_line(parsed)? {
                                tracing::debug!(?status, "ignoring unsolicited status line from babeld");
                            }
                        }
                        Err(err) => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("failed to parse babeld monitor output: {err}"),
                            )
                            .into());
                        }
                    }
                },
            }
        }
        Ok(())
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        let kill_res = if let Some(pid) = self.proc.id() {
            let pid: libc::pid_t = pid.try_into().expect("pid overflow");
            // SAFETY: pid >= 0, freshly checked.
            let rc = unsafe { libc::kill(pid, libc::SIGINT) };
            let rc_err = if rc != 0 && rc != libc::ESRCH {
                Err(io::Error::last_os_error().into())
            } else {
                Ok(())
            };
            let exit_code = async { Some(self.proc.wait().await) }
                .or(async {
                    // TODO: undocumented magic number
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    None
                })
                .await;
            match exit_code {
                Some(Ok(code)) => {
                    if code.success() {
                        rc_err
                    } else {
                        rc_err.and_then(|()| Err(BabbleError::BabeldCrashed(code.code())))
                    }
                }
                Some(Err(e)) => Err(e.into()),
                None => {
                    self.proc.kill().await?;
                    rc_err.and(Err(BabbleError::BabeldCrashed(None)))
                }
            }
        } else {
            Ok(())
        };
        let rem_res = match std::fs::remove_file(PRIVATE_SOCK_PATH) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        };
        kill_res.and(rem_res)
    }
}
