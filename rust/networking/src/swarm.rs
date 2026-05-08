//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::pin::Pin;

use futures_lite::Stream;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::info;
use zenoh::Result;
use zenoh::Session;
use zenoh::handlers::FifoChannelHandler;
use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::Subscriber;
use zenoh::sample::Sample;
use zenoh::sample::SampleKind;
use zerompk::{FromMessagePack, ToMessagePack};

#[derive(Debug)]
pub enum ToSwarm {
    Unsubscribe {
        topic: String,
        result_sender: oneshot::Sender<bool>,
    },
    Subscribe {
        topic: String,
        result_sender: oneshot::Sender<Result<bool>>,
    },
    Publish {
        topic: String,
        data: Vec<u8>,
        result_sender: oneshot::Sender<Result<()>>,
    },
}
#[derive(Debug, ToMessagePack, FromMessagePack)]
pub enum FromSwarm {
    Message { topic: String, data: Vec<u8> },
    Discovered {},
    Expired {},
}

pub type Topics = HashMap<String, Subscriber<()>>;
pub struct Swarm {
    cfg: zenoh::Config,
    from_client: mpsc::Receiver<ToSwarm>,
}

impl Swarm {
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = FromSwarm> + Send>> {
        let Swarm {
            cfg,
            mut from_client,
        } = self;
        let stream = async_stream::stream! {
            let (mut to_topics, mut from_topics) = mpsc::channel(1024);
            let mut topics = Topics::new();
            let Ok(mut session) = crate::open(cfg).await else { return; };
            let Ok((_token, discovery)) = register_liveness(&mut session).await else { return; };
            loop {
                tokio::select! {
                    msg = from_client.recv() => {
                        let Some(msg) = msg else { break };
                        on_message(&mut session, &mut topics, &mut to_topics, msg).await;
                    }
                    event = from_topics.recv() => {
                        if let Some(event) = event {
                            yield event
                        }
                    }
                    token = discovery.recv_async() => {
                        if let Ok(token) = token {
                            let key_expr = token.key_expr().as_str().to_owned();
                            let nid = key_expr.strip_prefix("nodes/").and_then(|s| s.strip_suffix("/live"));
                            yield match token.kind() {
                                SampleKind::Put => {
                                    info!("discovered: {nid:?}");
                                    FromSwarm::Discovered {}
                                }
                                SampleKind::Delete => {
                                    info!("expired: {nid:?}");
                                    FromSwarm::Expired {}
                                }
                            }
                        }

                    }
                }
            }
        };
        Box::pin(stream)
    }
}

async fn register_liveness(
    session: &mut Session,
) -> Result<(LivelinessToken, Subscriber<FifoChannelHandler<Sample>>)> {
    let token = session
        .liveliness()
        .declare_token(format!("nodes/{}/live", session.zid()))
        .await?;
    let sub = session
        .liveliness()
        .declare_subscriber("nodes/*/live")
        .history(true)
        .await?;
    Ok((token, sub))
}

async fn on_message(
    session: &mut Session,
    topics: &mut Topics,
    to_topics: &mut mpsc::Sender<FromSwarm>,
    msg: ToSwarm,
) {
    match msg {
        ToSwarm::Publish {
            topic,
            data,
            result_sender,
        } => {
            let res = session.put(format!("topics/{topic}"), data).await;
            _ = result_sender.send(res);
        }
        ToSwarm::Unsubscribe {
            topic,
            result_sender,
        } => {
            let Some((_, subscriber)) = topics.remove_entry(&topic) else {
                _ = result_sender.send(false);
                return;
            };
            _ = subscriber.undeclare().await;
            _ = result_sender.send(true);
        }
        ToSwarm::Subscribe {
            topic,
            result_sender,
        } => {
            assert!(topic.is_ascii());
            if topics.contains_key(&topic) {
                _ = result_sender.send(Ok(false));
                return;
            }
            let subscriber = match session
                .declare_subscriber(format!("topics/{topic}"))
                .allowed_origin(zenoh::sample::Locality::Remote)
                .callback({
                    let sender = to_topics.clone();
                    let topic = topic.clone();
                    move |sample| {
                        if sample.kind() != SampleKind::Put {
                            return;
                        }
                        _ = sender.try_send(FromSwarm::Message {
                            topic: topic.clone(),
                            data: sample.payload().to_bytes().to_vec(),
                        });
                    }
                })
                .await
            {
                Ok(p) => p,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };
            assert!(topics.insert(topic, subscriber).is_none());
            _ = result_sender.send(Ok(true));
        }
    }
}

pub fn create_swarm(
    identity: u128,
    from_client: mpsc::Receiver<ToSwarm>,
    bootstrap_peers: Vec<String>,
    listen_port: u16,
) -> Result<Swarm> {
    // todo: bootstrap
    if !bootstrap_peers.is_empty() || listen_port != 0 {
        todo!();
    }
    let cfg = crate::cfg(identity, 52414)?;
    let session = crate::open(cfg, 52414).await?;
    Ok(Swarm {
        session,
        from_client,
    })
}
