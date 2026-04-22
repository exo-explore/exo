use ipnet::Ipv6Net;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

use crate::Result;

pub mod command;
pub mod line;
pub mod runtime;
pub mod state;

use runtime::BabelRuntime;

/// An EUI-64 type aliased to [`macaddr::MacAddr8`].
pub type Eui64 = macaddr::MacAddr8;
pub use state::BabelState;

#[derive(Debug)]
pub enum Babble {
    AddIface(Box<str>),
}

#[tracing::instrument(skip(state_send, recv))]
pub async fn babel(
    advertised: Ipv6Net,
    mut recv: mpsc::Receiver<Babble>,
    state_send: watch::Sender<Arc<BabelState>>,
) -> Result<()> {
    // cannot spawn babel without interface first, so this spins until an interface found
    let iface = loop {
        match recv.recv().await {
            Some(Babble::AddIface(iface)) => break iface,
            None => return Ok(()),
        }
    };

    let mut runtime = BabelRuntime::spawn(advertised, &iface, state_send).await?;
    let res1 = runtime.run(recv).await;
    let res2 = runtime.shutdown().await;
    res1.and(res2)
}
