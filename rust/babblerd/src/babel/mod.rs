use ipnet::Ipv6Net;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc};

use crate::Result;

pub mod command;
pub mod line;
pub mod process;
pub mod session;

use process::BabeldProcess;
use session::BabelSession;

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
    AddIface(String),
}

#[tracing::instrument(skip(send, recv))]
pub async fn babel(
    my_range: Ipv6Net,
    mut recv: mpsc::Receiver<Babble>,
    send: broadcast::Sender<String>,
) -> Result<()> {
    // cannot spawn babel without interface first, so this spins until an interface found
    let iface = loop {
        match recv.recv().await {
            Some(Babble::AddIface(iface)) => break iface,
            None => return Ok(()),
        }
    };

    let babel = BabeldProcess::spawn(my_range, iface).await?;
    let session = BabelSession::connect(send).await?;
    let res1 = session.run(recv).await;
    let res2 = babel.shutdown().await;
    res1.and(res2)
}
