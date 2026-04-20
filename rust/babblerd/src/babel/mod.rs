use ipnet::Ipv6Net;
use std::fs::Permissions;
use std::io;
use std::os::unix::fs::PermissionsExt;
use tokio::time::Duration;

use futures_lite::FutureExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc};

use crate::{BabbleError, Result};

pub mod line;

#[tracing::instrument(skip_all)]
pub async fn handle_listener(sock: UnixStream, mut receiver: broadcast::Receiver<String>) {
    tracing::info!("new socket conn");
    let (reader, mut write) = sock.into_split();
    let mut reader = BufReader::new(reader).lines();
    loop {
        tokio::select! {
            read = reader.next_line() => {
                let Ok(Some(_)) = read else { break; };
            }
            recv = receiver.recv() => {
                let res = match recv {
                    Err(broadcast::error::RecvError::Lagged(_)) => { tracing::warn!("receiver lagged"); continue; },
                    Err(broadcast::error::RecvError::Closed) => { tracing::debug!("receiver closed, dropping connection"); break; },
                    Ok(s) => write.write_all(format!("{s}\n").as_bytes()).await,
                };
                if let Err(e) = res { tracing::warn!(error=%e, "failed to write to socket"); break; }
            }
        };
    }
    tracing::info!("closing socket conn");
    _ = write.shutdown().await;
}

#[cfg(target_os = "macos")]
const PRIVATE_SOCK_PATH: &str = "/var/run/babbler/private/babeld.sock";
#[cfg(target_os = "linux")]
const PRIVATE_SOCK_PATH: &str = "/run/babbler/private/babeld.sock";
#[cfg(target_os = "macos")]
const PRIVATE_DIR: &str = "/var/run/babbler/private";
#[cfg(target_os = "linux")]
const PRIVATE_DIR: &str = "/run/babbler/private";

#[derive(Debug)]
pub enum Babble {
    AddIface(String),
}

pub struct BabeldProcess {
    proc: tokio::process::Child,
}
impl Drop for BabeldProcess {
    #[inline]
    fn drop(&mut self) {
        // emergency sigkill babeld process to prevent leakage
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
impl BabeldProcess {
    #[tracing::instrument]
    async fn spawn(advertised: Ipv6Net, iface: String) -> Result<Self> {
        tokio::fs::create_dir_all(PRIVATE_DIR).await?;
        tokio::fs::set_permissions(PRIVATE_DIR, Permissions::from_mode(0o0700)).await?;
        tracing::info!("spawning babeld socket in {PRIVATE_SOCK_PATH}");
        let res = match Command::new("babeld")
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
            Ok(proc) => Ok(Self { proc }),
            Err(e) => {
                tracing::warn!(error=%e, "failed to spawn babeld");
                Err(e.into())
            }
        };

        tokio::time::sleep(Duration::from_millis(10)).await;
        while !matches!(tokio::fs::try_exists(PRIVATE_SOCK_PATH).await, Ok(true)) {
            tracing::info!("where is the sock");
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        res
    }

    #[tracing::instrument(skip(read, write, send))]
    async fn query(
        read: &mut Lines<BufReader<OwnedReadHalf>>,
        write: &mut OwnedWriteHalf,
        send: &broadcast::Sender<String>,
        cmd: &str,
    ) -> io::Result<Option<bool>> {
        write.write_all(cmd.as_bytes()).await?;
        loop {
            let Some(line) = read.next_line().await? else {
                tracing::warn!("babeld closed unexpectedly");
                return Ok(None);
            };
            tracing::info!("[babel] {:?}", line);

            // TODO: replace later with propper parsing
            match line::parse::parse_line(&line) {
                Ok(line) => tracing::info!("[parsed] {:?}", line),
                Err(err) => tracing::error!(error=%err, "failed to parse babeld line"),
            }

            let ret = match line.as_str() {
                "ok" => Ok(Some(true)),
                "bad" => {
                    tracing::warn!("malformed message sent to babeld");
                    Ok(Some(false))
                }
                _ if line.starts_with("no") => {
                    tracing::warn!("message rejected");
                    Ok(Some(false))
                }
                _ => Ok(None),
            };
            let Ok(_) = send.send(line) else {
                return Ok(None);
            };
            if !matches!(ret, Ok(None)) {
                return ret;
            }
        }
    }

    #[tracing::instrument(skip_all)]
    async fn supervise(
        &self,
        mut recv: mpsc::Receiver<Babble>,
        send: broadcast::Sender<String>,
    ) -> Result<()> {
        std::fs::set_permissions(PRIVATE_SOCK_PATH, Permissions::from_mode(0o0600))?;
        tracing::debug!("connecting to babeld.sock");
        let (reader, mut writer) = UnixStream::connect(PRIVATE_SOCK_PATH).await?.into_split();
        let mut babel_lines = BufReader::new(reader).lines();
        while let Some(s) = babel_lines.next_line().await? {
            tracing::debug!("[babeld] {}", s);
            if s == "ok" {
                break;
            }
        }
        tracing::info!("babeld ok");
        /* TODO(evan): push rather than pull
        if Self::query(&mut babel_lines, &mut writer, "monitor\n")
            .await?
            .is_none()
        {
            return Ok(());
        };
        */

        let mut interval = tokio::time::interval(Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if Self::query(&mut babel_lines, &mut writer, &send, "dump\n")
                        .await?
                        .is_none()
                    {
                        return Ok(());
                    }
                },
                babble = recv.recv() => {
                    tracing::debug!("[babble] {:?}", babble);
                    let Some(babble) = babble else {
                        break;
                    };
                    match babble {
                        Babble::AddIface(iface) => {
                            Self::query(&mut babel_lines, &mut writer, &send, format!("interface {iface}\n").as_ref()).await?;
                        }
                    }
                },
                line = babel_lines.next_line() => {
                    let Ok(Some(line)) = line else {
                        break;
                    };
                    tracing::debug!("[babeld] {}", line);

                    // TODO: replace later with propper parsing
                    match line::parse::parse_line(&line) {
                        Ok(line) => tracing::info!("[parsed] {:?}", line),
                        Err(err) => tracing::error!(error=%err, "failed to parse babeld line"),
                    }

                    let Ok(_) = send.send(line) else {
                        break;
                    };
                },
            }
        }
        Ok(())
    }

    async fn shutdown(mut self) -> Result<()> {
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

#[tracing::instrument(skip(send, recv))]
pub async fn babel(
    my_range: Ipv6Net,
    mut recv: mpsc::Receiver<Babble>,
    send: broadcast::Sender<String>,
) -> Result<()> {
    let iface = loop {
        match recv.recv().await {
            Some(Babble::AddIface(iface)) => {
                break iface;
            }
            None => return Ok(()),
        }
    };

    let babel = BabeldProcess::spawn(my_range, iface).await?;
    let res1 = babel.supervise(recv, send).await;
    let res2 = babel.shutdown().await;
    res1.and(res2)
}
