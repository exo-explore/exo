//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::pin::Pin;

use futures_lite::Stream;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use zenoh::Result;
use zenoh::Session;
use zenoh::handlers::FifoChannelHandler;
use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::Publisher;
use zenoh::pubsub::Subscriber;
use zenoh::qos::CongestionControl;
use zenoh::sample::Sample;
use zenoh::sample::SampleKind;

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
#[derive(Debug)]
pub enum FromSwarm {
    Message { topic: String, data: Vec<u8> },
    Discovered {},
    Expired {},
}

pub type Topics = HashMap<String, (Subscriber<()>, Publisher<'static>)>;
pub struct Swarm {
    pub session: crate::Session,
    pub from_client: mpsc::Receiver<ToSwarm>,
}

impl Swarm {
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = FromSwarm> + Send>> {
        let Swarm {
            session,
            mut from_client,
        } = self;
        let stream = async_stream::stream! {
            let mut session = session;
            let (mut to_topics, mut from_topics) = mpsc::channel(1024);
            let mut topics = Topics::new();
            let Ok((_token, discovery)) = register_liveness(&mut session.z).await else { return; };
            loop {
                tokio::select! {
                    msg = from_client.recv() => {
                        let Some(msg) = msg else { break };
                        on_message(&mut session.z, &mut topics, &mut to_topics, msg).await;
                    }
                    event = from_topics.recv() => {
                        if let Some(event) = event {
                            yield event
                        }
                    }
                    token = discovery.recv_async() => {
                        if let Ok(token) = token {
                            let key_expr = token.key_expr().as_str().to_owned();
                            let zid = key_expr.strip_prefix("live/");
                            yield match token.kind() {
                                SampleKind::Put => {
                                    log::info!("discovered: {zid:?}");
                                    FromSwarm::Discovered {}
                                }
                                SampleKind::Delete => {
                                    log::info!("expired: {zid:?}");
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
        .declare_token(format!("live/{}", session.zid()))
        .await?;
    let sub = session
        .liveliness()
        .declare_subscriber("live/*")
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
            let res = match topics.get(&topic) {
                Some(topic) => topic.1.put(data).await,
                None => {
                    // TODO: this should be an error but the python FromSwarm is somewhat nondeterministic
                    Ok(()) //Err("not subscribed to topic!".into()),
                }
            };
            _ = result_sender.send(res);
        }
        ToSwarm::Unsubscribe {
            topic,
            result_sender,
        } => {
            let Some((_, (subscriber, publisher))) = topics.remove_entry(&topic) else {
                _ = result_sender.send(false);
                return;
            };
            _ = publisher.undeclare().await;
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

            let publisher_res = session
                .declare_publisher(format!("topics/{topic}"))
                .congestion_control(CongestionControl::Block)
                .await;
            let publisher = match publisher_res {
                Ok(p) => p,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            let subscriber_res = session
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
                .await;
            let subscriber = match subscriber_res {
                Ok(s) => s,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            assert!(topics.insert(topic, (subscriber, publisher)).is_none());
            _ = result_sender.send(Ok(true));
        }
    }
}

pub async fn create_swarm(
    identity: &str,
    namespace: &str,
    from_client: mpsc::Receiver<ToSwarm>,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Swarm> {
    let cfg = crate::cfg(identity, listen_port)?;
    let session = crate::open(cfg, namespace, listen_port, discovery_service_port).await?;
    Ok(Swarm {
        session,
        from_client,
    })
}
