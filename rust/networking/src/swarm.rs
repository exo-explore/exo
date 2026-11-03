//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures_lite::Stream;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use zenoh::Result;
use zenoh::Session;
use zenoh::Wait;
use zenoh::config::ZenohId;
use zenoh::handlers::FifoChannelHandler;
use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::Publisher;
use zenoh::pubsub::Subscriber;
use zenoh::qos::{CongestionControl, Priority};
use zenoh::query::Queryable;
use zenoh::sample::Sample;
use zenoh::sample::SampleKind;

use crate::membership::{ALIVE_KEY_PREFIX, Change, Membership, PROBE_TIMEOUT};

#[derive(Debug)]
pub enum ToSwarm {
    Unsubscribe {
        topic: String,
        result_sender: oneshot::Sender<bool>,
    },
    Subscribe {
        topic: String,
        /// Send ahead of normal-priority traffic.
        high_priority: bool,
        result_sender: oneshot::Sender<Result<bool>>,
    },
    Publish {
        topic: String,
        data: Vec<u8>,
        result_sender: oneshot::Sender<Result<()>>,
    },
}
/// Messages received from the network wait here until exo reads them, and are dropped if it is
/// full (retries recover them, seconds later). It has to absorb the pauses of a busy node: at the
/// ~1,000 events/s of a loaded cluster this is about 15 s of messages.
const INCOMING_MESSAGE_CAPACITY: usize = 16_384;

#[derive(Debug)]
pub enum FromSwarm {
    Message {
        topic: String,
        data: Vec<u8>,
    },
    /// A node joined the cluster (or this node started)
    Discovered {
        zid: ZenohId,
    },
    /// A node left the cluster
    Expired {
        zid: ZenohId,
    },
}

impl From<Change> for FromSwarm {
    fn from(change: Change) -> Self {
        match change {
            Change::Joined(zid) => Self::Discovered { zid },
            Change::Left(zid) => Self::Expired { zid },
        }
    }
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
            let (mut to_topics, mut from_topics) = mpsc::channel(INCOMING_MESSAGE_CAPACITY);
            let mut topics = Topics::new();
            let Ok((_token, discovery)) = register_liveness(&mut session.z).await else { return; };
            let Ok((_alive, connections)) = watch_connections(&session.z).await else { return; };
            let (probe_results, mut finished_probes) = mpsc::unbounded_channel::<(ZenohId, bool)>();
            let mut membership = Membership::default();
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
                        let Ok(token) = token else { continue };
                        let key_expr = token.key_expr().as_str().to_owned();
                        let Some(zid) = key_expr.strip_prefix("live/").and_then(|zid| zid.parse::<ZenohId>().ok()) else {
                            log::warn!("ignoring liveliness token {key_expr}");
                            continue;
                        };
                        let change = match token.kind() {
                            SampleKind::Put => {
                                log::info!("discovered: {zid}");
                                membership.token_declared(zid)
                            }
                            SampleKind::Delete => {
                                log::info!("expired: {zid}");
                                membership.token_deleted(zid)
                            }
                        };
                        if let Some(change) = change {
                            yield change.into()
                        }
                    }
                    event = connections.recv_async() => {
                        let Ok(event) = event else { continue };
                        let zid = *event.transport().zid();
                        match event.kind() {
                            SampleKind::Put => {
                                log::info!("connected to {zid}");
                                if let Some(change) = membership.connection_opened(zid) {
                                    yield change.into()
                                }
                            }
                            SampleKind::Delete => {
                                log::info!("connection to {zid} closed");
                                if membership.connection_closed(zid) {
                                    let session = session.z.clone();
                                    let probe_results = probe_results.clone();
                                    tokio::spawn(async move {
                                        let answered = answers(&session, zid).await;
                                        _ = probe_results.send((zid, answered));
                                    });
                                }
                            }
                        }
                    }
                    Some((zid, answered)) = finished_probes.recv() => {
                        if let Some(change) = membership.probe_finished(zid, answered) {
                            log::info!(
                                "expired: {zid} (its connection closed and it didn't answer within {PROBE_TIMEOUT:?})"
                            );
                            yield change.into()
                        } else if answered {
                            log::info!("{zid} is still reachable after its connection closed");
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

type ConnectionEvents =
    zenoh::session::TransportEventsListener<FifoChannelHandler<zenoh::session::TransportEvent>>;

/// Answers whether this node is still there (see `membership`), and reports this node's
/// connections opening and closing.
async fn watch_connections(session: &Session) -> Result<(Queryable<()>, ConnectionEvents)> {
    let alive = session
        .declare_queryable(format!("{ALIVE_KEY_PREFIX}/{}", session.zid()))
        .callback(|query| {
            if let Err(e) = query
                .reply(query.key_expr().clone(), Vec::<u8>::new())
                .wait()
            {
                log::debug!("couldn't answer an alive query: {e}");
            }
        })
        .await?;
    let connections = session
        .info()
        .transport_events_listener()
        .history(true)
        .await?;
    Ok((alive, connections))
}

/// Whether a node answers, within PROBE_TIMEOUT, that it is still there.
async fn answers(session: &Session, zid: ZenohId) -> bool {
    let replies = session
        .get(format!("{ALIVE_KEY_PREFIX}/{zid}"))
        .priority(Priority::InteractiveHigh)
        .timeout(PROBE_TIMEOUT)
        .await;
    let Ok(replies) = replies else { return false };
    matches!(replies.recv_async().await, Ok(reply) if reply.result().is_ok())
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
            high_priority,
            result_sender,
        } => {
            assert!(topic.is_ascii());
            if topics.contains_key(&topic) {
                _ = result_sender.send(Ok(false));
                return;
            }

            // Drop rather than block when a peer's queue is full. With `Block`, one peer that
            // stops reading (a stalled process, a sleeping laptop) blocks every put on the topic,
            // and with it this whole loop, so no node receives anything from us. Receivers
            // already drop messages they can't keep up with, and the protocols on top recover
            // lost ones (event NACKs and resends, retried NACKs, repeated election rounds).
            let publisher_res = session
                .declare_publisher(format!("topics/{topic}"))
                .congestion_control(CongestionControl::Drop)
                .priority(if high_priority {
                    Priority::InteractiveHigh
                } else {
                    Priority::Data
                })
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
                    let dropped = Arc::new(AtomicU64::new(0));
                    move |sample| {
                        if sample.kind() != SampleKind::Put {
                            return;
                        }
                        let message = FromSwarm::Message {
                            topic: topic.clone(),
                            data: sample.payload().to_bytes().to_vec(),
                        };
                        if let Err(mpsc::error::TrySendError::Full(_)) = sender.try_send(message) {
                            // Logged at the 1st, 2nd, 4th, 8th... drop, so a flood shows without
                            // flooding the log
                            let total = dropped.fetch_add(1, Ordering::Relaxed) + 1;
                            if total.is_power_of_two() {
                                log::warn!(
                                    "dropped {total} incoming messages on topics/{topic} so far: exo isn't reading them fast enough"
                                );
                            }
                        }
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

#[cfg(test)]
mod tests {
    //! A node that crashes must be reported as gone by every other node's swarm, also when
    //! zenoh keeps its liveliness token (it does in clusters of three or more: see
    //! `membership`). The test binary runs itself as each node, in separate processes, so a
    //! node can be killed the way a crash kills it.
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::process::{Child, Command, Stdio};
    use std::sync::Arc;
    use std::sync::mpsc as std_mpsc;
    use std::time::{Duration, Instant};

    use futures_lite::StreamExt as _;
    use zenoh::config::{Locator, ZenohId};
    use zenoh_plugin_storage_manager::StoragesPlugin;
    use zenoh_plugin_trait::PluginsManager;

    use super::{FromSwarm, Swarm};
    use crate::membership::PROBE_TIMEOUT;

    const NODE_ENV: &str = "EXO_SWARM_TEST_NODE";
    const TEST_NAME: &str = "swarm::tests::every_node_reports_a_crashed_node_as_gone";
    const MESH_TIMEOUT: Duration = Duration::from_secs(30);

    #[test]
    fn every_node_reports_a_crashed_node_as_gone() {
        if let Ok(spec) = std::env::var(NODE_ENV) {
            run_node(&spec);
            return;
        }
        let count = 3;
        let Some(ports) = free_ports(count) else {
            eprintln!("skipping: no IPv6 loopback to run nodes on");
            return;
        };
        let zids: Vec<String> = (0..count)
            .map(|_| format!("{:x}", rand::random::<u128>() | (1 << 124)))
            .collect();
        let (events_tx, events) = std_mpsc::channel::<(usize, String, String)>();
        let mut children: Vec<Child> = (0..count)
            .map(|i| {
                let peers: Vec<String> = (0..count)
                    .filter(|&j| j != i)
                    .map(|j| format!("{}@{}", zids[j], ports[j]))
                    .collect();
                let mut child = Command::new(std::env::current_exe().expect("test binary"))
                    .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
                    .env(
                        NODE_ENV,
                        format!("{}@{} {}", zids[i], ports[i], peers.join(" ")),
                    )
                    // Events go to stderr: libtest's own "test <name> ..." line on stdout
                    // isn't ended before the test's output
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("start a node");
                let stderr = child.stderr.take().expect("node stderr");
                let events_tx = events_tx.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        let mut words = line.split_whitespace();
                        if words.next() != Some("SWARM") {
                            continue;
                        }
                        if let (Some(kind), Some(zid)) = (words.next(), words.next()) {
                            let _ = events_tx.send((i, kind.to_owned(), zid.to_owned()));
                        }
                    }
                });
                child
            })
            .collect();

        let mut joined: Vec<HashSet<String>> = vec![HashSet::new(); count];
        let deadline = Instant::now() + MESH_TIMEOUT;
        while joined.iter().any(|nodes| nodes.len() < count) {
            match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok((node, kind, zid)) if kind == "joined" => {
                    joined[node].insert(zid);
                }
                Ok(_) => {}
                Err(_) => {
                    kill_all(&mut children);
                    panic!("the nodes never all saw each other: {joined:?}");
                }
            }
        }

        let victim = 1;
        children[victim].kill().expect("kill a node");
        let mut reported: HashSet<usize> = HashSet::new();
        let deadline = Instant::now() + PROBE_TIMEOUT + Duration::from_secs(3);
        while reported.len() < count - 1 {
            match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok((node, kind, zid)) if kind == "left" && zid == zids[victim] => {
                    reported.insert(node);
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        kill_all(&mut children);
        assert_eq!(
            reported.len(),
            count - 1,
            "only nodes {reported:?} reported the killed node {victim} as gone"
        );
    }

    fn kill_all(children: &mut [Child]) {
        for child in children.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Ports below the ephemeral range that nothing listens on, on the IPv6 loopback
    fn free_ports(count: usize) -> Option<Vec<u16>> {
        TcpListener::bind("[::1]:0").ok()?;
        let base = 20_000 + rand::random::<u16>() % 20_000;
        let ports: Vec<u16> = (base..)
            .filter(|&port| {
                TcpListener::bind(("::1", port)).is_ok() && TcpListener::bind(("::", port)).is_ok()
            })
            .take(count)
            .collect();
        Some(ports)
    }

    /// A node: exo's swarm on an exo-configured router that dials its peers the way discovery
    /// does (only those with a smaller id), reporting who joins and leaves.
    fn run_node(spec: &str) {
        let mut parts = spec.split_whitespace();
        let (zid, port) = parse_peer(parts.next().expect("own id and port"));
        let peers: Vec<(String, u16)> = parts.map(parse_peer).collect();
        tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(async move {
                let cfg = crate::cfg(&zid, port).expect("config");
                let mut plugins = PluginsManager::static_plugins_only();
                plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
                let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
                    .plugins_manager(plugins)
                    .build()
                    .await
                    .expect("zenoh runtime");
                let z = zenoh::session::init(runtime.clone().into())
                    .await
                    .expect("zenoh session");
                runtime.start().await.expect("start zenoh");
                let own: ZenohId = z.zid();
                let dialer = tokio::spawn(async move {
                    loop {
                        for (peer, peer_port) in &peers {
                            let peer: ZenohId = peer.parse().expect("peer id");
                            if peer > own {
                                continue;
                            }
                            let locator = Locator::new("tcp", format!("[::1]:{peer_port}"), "")
                                .expect("locator");
                            let runtime = runtime.clone();
                            tokio::spawn(async move {
                                runtime.connect_peer(&peer.into(), &[locator]).await;
                            });
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                });
                let session = crate::Session {
                    z,
                    _jh: Arc::new(crate::AbortOnDrop(dialer)),
                };
                let (_to_swarm, from_client) = tokio::sync::mpsc::channel(16);
                let mut stream = Swarm {
                    session,
                    from_client,
                }
                .into_stream();
                while let Some(event) = stream.next().await {
                    match event {
                        FromSwarm::Discovered { zid } => eprintln!("SWARM joined {zid}"),
                        FromSwarm::Expired { zid } => eprintln!("SWARM left {zid}"),
                        FromSwarm::Message { .. } => {}
                    }
                }
            });
    }

    fn parse_peer(peer: &str) -> (String, u16) {
        let (zid, port) = peer.split_once('@').expect("ID@PORT");
        (zid.to_owned(), port.parse().expect("port"))
    }
}
