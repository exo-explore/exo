//! Live local-socket session with the managed `babeld` process.
//!
//! This module owns the runtime control connection to `babeld`'s private Unix socket:
//!
//! - connecting to the socket after the child has started
//! - sending typed [`crate::babel::command::BabelCommand`] values
//! - parsing inbound lines into [`crate::babel::line::BabelLine`] values for logging/control
//! - correlating terminal [`crate::babel::line::Status`] lines with commands
//! - running the current dump-based supervision loop
//!
//! It intentionally does **not** own:
//!
//! - child-process spawn/shutdown, which belongs in [`crate::babel::process`]
//! - long-lived reduced state, which should eventually live in a separate state layer

use std::io;
use std::os::unix::fs::PermissionsExt;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc};
use tokio::time::Duration;

use crate::babel::command::BabelCommand;
use crate::babel::line::{self, BabelLine, Status};
use crate::babel::process::private_sock_path;
use crate::babel::Babble;
use crate::Result;

pub(crate) struct BabelSession {
    read: Lines<BufReader<OwnedReadHalf>>,
    write: OwnedWriteHalf,
    send: broadcast::Sender<String>,
}

impl BabelSession {
    #[tracing::instrument(skip(send))]
    pub(crate) async fn connect(send: broadcast::Sender<String>) -> Result<Self> {
        // TODO: magic undocumented number
        std::fs::set_permissions(private_sock_path(), std::fs::Permissions::from_mode(0o0600))?;
        tracing::debug!("connecting to babeld.sock");
        let (reader, write) = UnixStream::connect(private_sock_path()).await?.into_split();
        Ok(Self {
            read: BufReader::new(reader).lines(),
            write,
            send,
        })
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn await_ready(&mut self) -> Result<()> {
        /// TODO: replace with parsing logic which we have anyways...
        ///       also i just don't like this being its own method for some reason
        while let Some(line) = self.read.next_line().await? {
            tracing::debug!("[babeld] {}", line);
            if line == "ok" {
                tracing::info!("babeld ok");
                break;
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn query(&mut self, cmd: &BabelCommand) -> io::Result<Option<Status>> {
        self.write.write_all(cmd.encode().as_bytes()).await?;
        loop {
            let Some(line) = self.read.next_line().await? else {
                tracing::warn!("babeld closed unexpectedly");
                return Ok(None);
            };
            let status = self.observe_line(line)?;
            if let Some(status) = status {
                match &status {
                    Status::Ok => {}
                    Status::Bad => tracing::warn!("malformed message sent to babeld"),
                    Status::No(rest) => tracing::warn!("message rejected: {rest:?}"),
                }
                return Ok(Some(status));
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn observe_line(&self, line: String) -> io::Result<Option<Status>> {
        tracing::info!("[babel] {:?}", line);

        let status = match line::parse::parse_line(&line) {
            Ok(parsed) => {
                tracing::info!("[parsed] {:?}", parsed);
                match parsed {
                    BabelLine::Status(status) => Some(status),
                    _ => None,
                }
            }
            Err(err) => {
                tracing::error!(error=%err, "failed to parse babeld line");
                None
            }
        };

        self.send
            .send(line)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "babel listeners dropped"))?;
        Ok(status)
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn run(mut self, mut recv: mpsc::Receiver<Babble>) -> Result<()> {
        self.await_ready().await?;

        /* TODO(evan): push rather than pull
        if self
            .query(&BabelCommand::Monitor)
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
                    if self.query(&BabelCommand::Dump).await?.is_none() {
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
                            let cmd = BabelCommand::Interface(iface.into_boxed_str());
                            self.query(&cmd).await?;
                        }
                    }
                },
                line = self.read.next_line() => {
                    let Ok(Some(line)) = line else {
                        break;
                    };
                    if self.observe_line(line)?.is_some() {
                        tracing::debug!("ignoring unsolicited status line from babeld");
                    }
                },
            }
        }
        Ok(())
    }
}
