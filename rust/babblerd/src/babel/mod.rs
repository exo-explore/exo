use ipnet::Ipv6Net;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc, watch};

use crate::Result;

pub mod command;
pub mod line;
pub mod runtime;
pub mod state;

use runtime::BabelRuntime;

/// An EUI-64 type aliased to [`macaddr::MacAddr8`].
pub type Eui64 = macaddr::MacAddr8;
pub use state::BabelState;

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

#[derive(Debug)]
pub enum Babble {
    AddIface(Box<str>),
}

#[tracing::instrument(skip(line_send, state_send, recv))]
pub async fn babel(
    my_range: Ipv6Net,
    mut recv: mpsc::Receiver<Babble>,
    line_send: broadcast::Sender<String>,
    state_send: watch::Sender<Arc<BabelState>>,
) -> Result<()> {
    // cannot spawn babel without interface first, so this spins until an interface found
    let iface = loop {
        match recv.recv().await {
            Some(Babble::AddIface(iface)) => break iface,
            None => return Ok(()),
        }
    };

    let mut runtime = BabelRuntime::spawn(my_range, &iface, line_send, state_send).await?;
    let res1 = runtime.run(recv).await;
    let res2 = runtime.shutdown().await;
    res1.and(res2)
}
