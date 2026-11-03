//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures_lite::Stream;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::MissedTickBehavior;
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

use crate::membership::{
    ALIVE_KEY_PREFIX, GRACE, HOLD_OFF, Member, Membership, Now, PROBE_FIRST_ASK_DELAY,
    PROBE_TIMEOUT, REJOIN, Snapshot, TICK,
};

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
    /// A node's state changed, as this node sees it. This node's own state comes first, when
    /// the swarm starts. Each change is reported once.
    Membership(Member),
}

/// This node's view of the cluster, readable at any time.
///
/// Python reads it (`NetworkingHandle.members()`), so nothing may log, call into Python or wait
/// while holding its lock: a log record can wait for Python's GIL, and a Python thread reading
/// the snapshot would then wait for the lock with the GIL held, for good. `with` keeps the lock
/// to the closure; log after it.
#[derive(Clone)]
pub struct Members {
    membership: Arc<Mutex<Membership>>,
    from_exo: mpsc::UnboundedSender<FromExo>,
}

/// What exo tells the membership, that only exo knows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FromExo {
    /// The coordinator removed the node from the cluster
    Removed(ZenohId),
}

impl Members {
    pub fn snapshot(&self) -> Snapshot {
        self.with(|membership| membership.snapshot())
    }

    /// Passes `input` on to the task that drives the membership: it never waits, and never
    /// takes the lock
    pub fn tell(&self, input: FromExo) {
        _ = self.from_exo.send(input);
    }

    fn with<T>(&self, update: impl FnOnce(&mut Membership) -> T) -> T {
        update(&mut self.membership.lock())
    }
}

pub type Topics = HashMap<String, (Subscriber<()>, Publisher<'static>)>;
pub struct Swarm {
    session: crate::Session,
    from_client: mpsc::Receiver<ToSwarm>,
    members: Members,
    from_exo: mpsc::UnboundedReceiver<FromExo>,
}

impl Swarm {
    pub fn new(session: crate::Session, from_client: mpsc::Receiver<ToSwarm>) -> Self {
        Self::with_grace(session, from_client, GRACE)
    }

    /// With `grace` instead of `GRACE` as how long a peer is Suspect before it is Gone
    pub fn with_grace(
        session: crate::Session,
        from_client: mpsc::Receiver<ToSwarm>,
        grace: Duration,
    ) -> Self {
        let membership = Membership::new(session.z.zid(), grace, Now::read());
        let (to_membership, from_exo) = mpsc::unbounded_channel();
        Self {
            session,
            from_client,
            members: Members {
                membership: Arc::new(Mutex::new(membership)),
                from_exo: to_membership,
            },
            from_exo,
        }
    }

    pub fn members(&self) -> Members {
        self.members.clone()
    }

    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = FromSwarm> + Send>> {
        let Swarm {
            session,
            mut from_client,
            members,
            from_exo,
        } = self;
        let stream = async_stream::stream! {
            let mut session = session;
            let (mut to_topics, mut from_topics) = mpsc::channel(INCOMING_MESSAGE_CAPACITY);
            let mut topics = Topics::new();
            let Ok(inputs) = MembershipInputs::declare(&mut session.z).await else { return; };
            // Membership runs on its own, so it keeps time (and notices this node's stalls)
            // however often exo reads this stream
            let (changes_sender, mut changes) = mpsc::unbounded_channel();
            let _membership = crate::AbortOnDrop(tokio::spawn(drive_membership(
                session.z.clone(),
                members,
                inputs,
                from_exo,
                changes_sender,
            )));
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
                    Some(change) = changes.recv() => {
                        yield FromSwarm::Membership(change)
                    }
                }
            }
        };
        Box::pin(stream)
    }
}

/// What membership learns from zenoh: this node's liveliness token and everyone's, the
/// queryable that answers that this node is still there, and this node's connections
struct MembershipInputs {
    _token: LivelinessToken,
    tokens: Subscriber<FifoChannelHandler<Sample>>,
    _alive: Queryable<()>,
    connections: ConnectionEvents,
}

impl MembershipInputs {
    async fn declare(session: &mut Session) -> Result<Self> {
        let (token, tokens) = register_liveness(session).await?;
        let (alive, connections) = watch_connections(session).await?;
        Ok(Self {
            _token: token,
            tokens,
            _alive: alive,
            connections,
        })
    }
}

/// Feeds membership its inputs and a tick, and sends on each change it reports
async fn drive_membership(
    session: Session,
    members: Members,
    inputs: MembershipInputs,
    mut from_exo: mpsc::UnboundedReceiver<FromExo>,
    changes: mpsc::UnboundedSender<Member>,
) {
    let MembershipInputs {
        _token,
        tokens,
        _alive,
        connections,
    } = inputs;
    let (probe_results, mut finished_probes) = mpsc::unbounded_channel::<(ZenohId, bool)>();
    let mut tick = tokio::time::interval(TICK);
    // After a stall, one late tick rather than a burst of them
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let own = members.with(|membership| membership.own());
    if changes.send(own).is_err() {
        return;
    }
    // Every log call below is outside `members.with` (see `Members`)
    loop {
        let reported: Vec<Member> = tokio::select! {
            // The tick first: after a stall it tells us so before any queued input is handled.
            // Then the inputs a node has few of (probe answers, connections) ahead of tokens, so
            // that a burst of tokens can't hold them up.
            biased;
            _ = tick.tick() => {
                let (changes, stalled_for) = members.with(|membership| {
                    (membership.tick(Now::read()), membership.take_stall())
                });
                if let Some(late) = stalled_for {
                    log::warn!(
                        "this node was unresponsive for {late:.0?}; resuming for {:?} before it \
                         times any lost peer",
                        HOLD_OFF + REJOIN
                    );
                }
                changes
            }
            Some((zid, answered)) = finished_probes.recv() => {
                let change = members
                    .with(|membership| membership.probe_finished(zid, answered, Now::read()));
                if answered {
                    log::debug!("{zid} answered that it is still there");
                } else if change.is_some() {
                    log::info!("{zid} didn't answer within {PROBE_TIMEOUT:?}");
                }
                change.into_iter().collect()
            }
            event = connections.recv_async() => {
                let Ok(event) = event else { return };
                let zid = *event.transport().zid();
                match event.kind() {
                    SampleKind::Put => {
                        log::info!("connected to {zid}");
                        members
                            .with(|membership| membership.connection_opened(zid, Now::read()))
                            .into_iter()
                            .collect()
                    }
                    SampleKind::Delete => {
                        log::info!("connection to {zid} closed");
                        members.with(|membership| membership.connection_closed(zid));
                        Vec::new()
                    }
                }
            }
            Some(input) = from_exo.recv() => {
                match input {
                    FromExo::Removed(zid) => {
                        log::info!("the coordinator removed {zid}");
                        members.with(|membership| membership.forget(zid));
                    }
                }
                Vec::new()
            }
            token = tokens.recv_async() => {
                // The session is closed
                let Ok(token) = token else { return };
                let key_expr = token.key_expr().as_str().to_owned();
                let Some(zid) = key_expr.strip_prefix("live/").and_then(|zid| zid.parse::<ZenohId>().ok()) else {
                    log::warn!("ignoring liveliness token {key_expr}");
                    continue;
                };
                let now = Now::read();
                let change = match token.kind() {
                    SampleKind::Put => {
                        log::info!("token of {zid} declared");
                        members.with(|membership| membership.token_put(zid, now))
                    }
                    SampleKind::Delete => {
                        log::info!("token of {zid} deleted");
                        members.with(|membership| membership.token_deleted(zid, now))
                    }
                };
                change.into_iter().collect()
            }
        };
        // Ask each node the membership wants asked, outside its lock
        for zid in members.with(Membership::take_probes) {
            let session = session.clone();
            let probe_results = probe_results.clone();
            tokio::spawn(async move {
                let answered = answers(&session, zid).await;
                _ = probe_results.send((zid, answered));
            });
        }
        for change in reported {
            log::info!("membership: {} is {:?}", change.node, change.state);
            if changes.send(change).is_err() {
                return;
            }
        }
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

/// Whether a node answers, within `PROBE_TIMEOUT`, that it is still there. It is first asked
/// after `PROBE_FIRST_ASK_DELAY`, and again after that delay each time a query ends without an
/// answer: until zenoh has recomputed its routes after the closed link, a query to a node that
/// is reachable only through another one finds no route and ends at once.
async fn answers(session: &Session, zid: ZenohId) -> bool {
    let deadline = tokio::time::Instant::now() + PROBE_TIMEOUT;
    loop {
        tokio::time::sleep_until(
            (tokio::time::Instant::now() + PROBE_FIRST_ASK_DELAY).min(deadline),
        )
        .await;
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        if left.is_zero() {
            return false;
        }
        let Ok(replies) = session
            .get(format!("{ALIVE_KEY_PREFIX}/{zid}"))
            .priority(Priority::InteractiveHigh)
            .timeout(left)
            .await
        else {
            continue;
        };
        if matches!(replies.recv_async().await, Ok(reply) if reply.result().is_ok()) {
            return true;
        }
    }
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
    Ok(Swarm::new(session, from_client))
}

#[cfg(test)]
mod tests {
    //! Every node's swarm must report what happens to a peer: a crash (Suspect at once, also
    //! when zenoh keeps the peer's liveliness token, as it does in clusters of three or more:
    //! see `membership`; then Gone after the grace), a freeze (Suspect after the lease, Alive
    //! again when it resumes), and a lost direct link to a peer that is still reachable through
    //! another node (it stays Alive). The test binary runs itself as each node, in separate
    //! processes, so a node can be killed or stopped the way a crash or a sleep does it.
    use std::collections::{HashMap, HashSet};
    use std::io::{BufRead, BufReader};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc as std_mpsc;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::{Duration, Instant};

    use futures_lite::StreamExt as _;
    use zenoh::config::{Locator, ZenohId};
    use zenoh::sample::SampleKind;
    use zenoh_plugin_storage_manager::StoragesPlugin;
    use zenoh_plugin_trait::PluginsManager;

    use super::{FromSwarm, Swarm};
    use crate::membership::{GRACE, LEASE, MemberState, PROBE_TIMEOUT, REPROBE_INTERVAL, TICK};

    const NODE_ENV: &str = "EXO_SWARM_TEST_NODE";
    /// Set to make these tests fail, instead of skipping, where there is no IPv6 loopback
    const REQUIRED_ENV: &str = "EXO_SWARM_TESTS_REQUIRED";
    const MESH_TIMEOUT: Duration = Duration::from_secs(30);
    /// For seeing the lines a node prints, and a tick (1 s) to pass
    const SLACK: Duration = Duration::from_secs(3);

    #[test]
    fn every_node_reports_a_crashed_node_as_gone() {
        const NAME: &str = "swarm::tests::every_node_reports_a_crashed_node_as_gone";
        if let Ok(spec) = std::env::var(NODE_ENV) {
            run_node(&spec);
            return;
        }
        let grace = Duration::from_secs(3);
        let Some(mut cluster) = Cluster::start(NAME, 3, grace) else {
            return;
        };
        let victim = 1;
        let observers = cluster.others(victim);
        let killed = Instant::now();
        cluster.kill(victim);

        let suspect = cluster.wait_for_all(
            &observers,
            victim,
            "Suspect(Disconnected)",
            killed,
            killed + PROBE_TIMEOUT + SLACK,
        );
        let gone = cluster.wait_for_all(
            &observers,
            victim,
            "Gone(Disconnected)",
            killed,
            killed + PROBE_TIMEOUT + grace + SLACK,
        );
        for node in observers {
            let after = gone[&node] - suspect[&node];
            eprintln!(
                "node {node}: suspect {:?} after the kill, gone {after:?} after that",
                suspect[&node] - killed
            );
            assert!(
                after >= grace - Duration::from_millis(200) && after <= grace + SLACK,
                "node {node} reported the crashed node gone {after:?} after it was suspect"
            );
        }
    }

    #[test]
    fn every_node_reports_a_frozen_node_as_suspect_then_alive_again() {
        const NAME: &str =
            "swarm::tests::every_node_reports_a_frozen_node_as_suspect_then_alive_again";
        if let Ok(spec) = std::env::var(NODE_ENV) {
            run_node(&spec);
            return;
        }
        let Some(mut cluster) = Cluster::start(NAME, 3, GRACE) else {
            return;
        };
        let victim = 1;
        let observers = cluster.others(victim);
        let stopped = Instant::now();
        cluster.signal(victim, "STOP");

        let suspect = cluster.wait_for_all(
            &observers,
            victim,
            "Suspect(Disconnected)",
            stopped,
            stopped + LEASE + PROBE_TIMEOUT + SLACK,
        );
        for (node, at) in &suspect {
            let after = *at - stopped;
            eprintln!("node {node}: suspect {after:?} after the freeze");
            // zenoh sends 4 keep-alives per lease, so the last one came up to a quarter of a
            // lease before the freeze
            assert!(
                after >= LEASE * 3 / 4 - Duration::from_millis(500),
                "node {node} reported the frozen node suspect only {after:?} after it froze"
            );
        }
        let first_suspect = suspect.values().min().copied().unwrap_or(stopped);

        let resumed = Instant::now();
        cluster.signal(victim, "CONT");
        let alive =
            cluster.wait_for_all(&observers, victim, "Alive", resumed, first_suspect + GRACE);
        let alive_again = alive.values().max().map(|at| *at - resumed);
        // It noticed its own stall
        cluster.wait_for_all(&[victim], victim, "Resuming", resumed, resumed + SLACK);
        assert!(
            cluster.reported("Gone").is_empty(),
            "nobody may be gone: {:?}",
            cluster.reported("Gone")
        );
        eprintln!(
            "suspect after {:?}, alive again {alive_again:?} after it resumed",
            first_suspect - stopped
        );
    }

    /// The direct connection between two nodes closes while both still reach each other
    /// through the third one: the probe must be answered through it (asked at once, before
    /// zenoh had recomputed its routes, it wasn't always), and so must the probes repeated while
    /// it has no connection of its own. Then that node crashes: nothing closes on the far end of
    /// the cut and zenoh may keep its token, but the repeated probe shows it, and it is Gone a
    /// grace later (it used to stay Alive for good).
    #[test]
    fn a_node_reached_through_another_node_stays_alive_until_it_crashes() {
        const NAME: &str =
            "swarm::tests::a_node_reached_through_another_node_stays_alive_until_it_crashes";
        if let Ok(spec) = std::env::var(NODE_ENV) {
            run_node(&spec);
            return;
        }
        let Some(ports) = loopback_ports(NAME, 4) else {
            return;
        };
        let zids = random_zids(3);
        // Nodes 0 and 1 are connected through a proxy (the one with the greater id dials)
        let ids: Vec<ZenohId> = zids.iter().map(|zid| zid.parse().expect("zid")).collect();
        let (dialer, dialed) = if ids[0] > ids[1] { (0, 1) } else { (1, 0) };
        let proxy_port = ports[3];
        let proxy = Proxy::start(proxy_port, ports[dialed]);
        let grace = Duration::from_secs(3);
        let mut cluster = Cluster::spawn(NAME, grace, &zids, &ports[..3], |from, to| {
            if from == dialer && to == dialed {
                proxy_port
            } else {
                ports[to]
            }
        });
        cluster.wait_for_mesh();

        let cut = Instant::now();
        proxy.cut();
        // Both ends see the direct connection close...
        cluster.wait_for_all(&[dialer], dialed, "closed", cut, cut + SLACK);
        cluster.wait_for_all(&[dialed], dialer, "closed", cut, cut + SLACK);
        // ... and neither takes the other for unreachable, also when asked again
        let asked_again = REPROBE_INTERVAL + TICK + PROBE_TIMEOUT;
        std::thread::sleep(asked_again + SLACK);
        let suspect = cluster.reported("Suspect");
        assert!(
            suspect.is_empty(),
            "nodes reported as unreachable though still reachable: {suspect:#?}"
        );

        let killed = Instant::now();
        cluster.kill(dialed);
        let suspect = cluster.wait_for_all(
            &[dialer],
            dialed,
            "Suspect(Disconnected)",
            killed,
            killed + asked_again + SLACK,
        );
        let gone = cluster.wait_for_all(
            &[dialer],
            dialed,
            "Gone(Disconnected)",
            killed,
            suspect[&dialer] + grace + SLACK,
        );
        eprintln!(
            "the node reached through another: suspect {:?} after it was killed, gone {:?} later",
            suspect[&dialer] - killed,
            gone[&dialer] - suspect[&dialer]
        );
    }

    /// A TCP proxy that can be cut: its connections are shut and it accepts no new ones
    struct Proxy {
        port: u16,
        cut: Arc<AtomicBool>,
        streams: Arc<StdMutex<Vec<TcpStream>>>,
    }

    impl Proxy {
        fn start(port: u16, target: u16) -> Self {
            let listener = TcpListener::bind(("::1", port)).expect("proxy listener");
            let cut = Arc::new(AtomicBool::new(false));
            let streams: Arc<StdMutex<Vec<TcpStream>>> = Arc::default();
            {
                let cut = Arc::clone(&cut);
                let streams = Arc::clone(&streams);
                std::thread::spawn(move || {
                    for client in listener.incoming() {
                        if cut.load(std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }
                        let Ok(client) = client else { continue };
                        let Ok(server) = TcpStream::connect(("::1", target)) else {
                            continue;
                        };
                        let (Ok(client_copy), Ok(server_copy)) =
                            (client.try_clone(), server.try_clone())
                        else {
                            continue;
                        };
                        streams
                            .lock()
                            .expect("proxy streams")
                            .extend([client_copy, server_copy]);
                        let (Ok(client_reader), Ok(server_writer)) =
                            (client.try_clone(), server.try_clone())
                        else {
                            continue;
                        };
                        for (mut from, mut to) in [(client_reader, server_writer), (server, client)]
                        {
                            std::thread::spawn(move || {
                                let _ = std::io::copy(&mut from, &mut to);
                                let _ = to.shutdown(Shutdown::Both);
                            });
                        }
                    }
                });
            }
            Self { port, cut, streams }
        }

        fn cut(&self) {
            self.cut.store(true, std::sync::atomic::Ordering::SeqCst);
            // Wake the accept loop, which then stops
            let _ = TcpStream::connect(("::1", self.port));
            for stream in self.streams.lock().expect("proxy streams").iter() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
    }

    /// Nodes running as child processes, and what each reports
    struct Cluster {
        children: Vec<Child>,
        zids: Vec<String>,
        events: std_mpsc::Receiver<Event>,
        /// Every event seen so far
        seen: Vec<Event>,
    }

    #[derive(Debug, Clone)]
    struct Event {
        /// Which node reported it
        node: usize,
        /// The state it reported, e.g. `Suspect(Disconnected)`, or `closed` for a connection
        state: String,
        /// About which node
        about: String,
        at: Instant,
    }

    impl Cluster {
        /// Starts `count` nodes, each dialing the others directly, and waits until each
        /// reports every node (itself included) Alive. `None` when there is no IPv6 loopback.
        fn start(test_name: &str, count: usize, grace: Duration) -> Option<Self> {
            let ports = loopback_ports(test_name, count)?;
            let zids = random_zids(count);
            let mut cluster = Self::spawn(test_name, grace, &zids, &ports, |_, to| ports[to]);
            cluster.wait_for_mesh();
            Some(cluster)
        }

        /// Starts a node for each of `zids`, listening on `ports`; node `from` dials node
        /// `to` on `port_of(from, to)`
        fn spawn(
            test_name: &str,
            grace: Duration,
            zids: &[String],
            ports: &[u16],
            port_of: impl Fn(usize, usize) -> u16,
        ) -> Self {
            let count = zids.len();
            let (events_tx, events) = std_mpsc::channel::<Event>();
            let children: Vec<Child> = (0..count)
                .map(|i| {
                    let peers: Vec<String> = (0..count)
                        .filter(|&j| j != i)
                        .map(|j| format!("{}@{}", zids[j], port_of(i, j)))
                        .collect();
                    let mut child = Command::new(std::env::current_exe().expect("test binary"))
                        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
                        .env(
                            NODE_ENV,
                            format!(
                                "{} {}@{} {}",
                                grace.as_millis(),
                                zids[i],
                                ports[i],
                                peers.join(" ")
                            ),
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
                            if let (Some(state), Some(about)) = (words.next(), words.next()) {
                                let _ = events_tx.send(Event {
                                    node: i,
                                    state: state.to_owned(),
                                    about: about.to_owned(),
                                    at: Instant::now(),
                                });
                            }
                        }
                    });
                    child
                })
                .collect();
            Self {
                children,
                zids: zids.to_vec(),
                events,
                seen: Vec::new(),
            }
        }

        /// Waits until every node reports every node (itself included) Alive
        fn wait_for_mesh(&mut self) {
            let all: Vec<usize> = (0..self.zids.len()).collect();
            let started = Instant::now();
            for node in 0..self.zids.len() {
                self.wait_for_all(&all, node, "Alive", started, started + MESH_TIMEOUT);
            }
        }

        fn others(&self, node: usize) -> Vec<usize> {
            (0..self.zids.len()).filter(|&i| i != node).collect()
        }

        fn kill(&mut self, node: usize) {
            self.children[node].kill().expect("kill a node");
        }

        fn signal(&self, node: usize, signal: &str) {
            let status = Command::new("kill")
                .args([format!("-{signal}"), self.children[node].id().to_string()])
                .status()
                .expect("run kill");
            assert!(status.success(), "kill -{signal} failed");
        }

        /// Waits until each of `nodes` has reported `state` for node `about` at `after` or
        /// later, and returns when each first did. Panics at `deadline`.
        fn wait_for_all(
            &mut self,
            nodes: &[usize],
            about: usize,
            state: &str,
            after: Instant,
            deadline: Instant,
        ) -> HashMap<usize, Instant> {
            let about_zid = self.zids[about].clone();
            let matches = |event: &Event| {
                event.about == about_zid
                    && event.state == state
                    && nodes.contains(&event.node)
                    && event.at >= after
            };
            let mut reported: HashMap<usize, Instant> = HashMap::new();
            for event in self.seen.iter().filter(|event| matches(event)) {
                reported.entry(event.node).or_insert(event.at);
            }
            loop {
                if nodes.iter().all(|node| reported.contains_key(node)) {
                    return reported;
                }
                match self
                    .events
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                {
                    Ok(event) => {
                        if matches(&event) {
                            reported.entry(event.node).or_insert(event.at);
                        }
                        self.seen.push(event);
                    }
                    Err(_) => {
                        let missing: HashSet<usize> = nodes
                            .iter()
                            .copied()
                            .filter(|node| !reported.contains_key(node))
                            .collect();
                        let seen = self.seen.clone();
                        self.stop();
                        panic!(
                            "nodes {missing:?} never reported node {about} {state}; saw {seen:#?}"
                        );
                    }
                }
            }
        }

        /// Every event so far whose state starts with `state`
        fn reported(&mut self, state: &str) -> Vec<Event> {
            self.seen.extend(self.events.try_iter());
            self.seen
                .iter()
                .filter(|event| event.state.starts_with(state))
                .cloned()
                .collect()
        }

        fn stop(&mut self) {
            for child in &mut self.children {
                let _ = Command::new("kill")
                    .args(["-CONT", &child.id().to_string()])
                    .status();
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    impl Drop for Cluster {
        fn drop(&mut self) {
            self.stop();
        }
    }

    fn random_zids(count: usize) -> Vec<String> {
        (0..count)
            .map(|_| format!("{:x}", rand::random::<u128>() | (1 << 124)))
            .collect()
    }

    /// `count` free ports on the IPv6 loopback. Without one (some build sandboxes) the test is
    /// skipped, saying so, unless `EXO_SWARM_TESTS_REQUIRED` is set, when it fails.
    fn loopback_ports(test_name: &str, count: usize) -> Option<Vec<u16>> {
        let ports = free_ports(count);
        if ports.is_none() {
            let message = format!("SKIPPED {test_name}: no IPv6 loopback to run nodes on");
            assert!(std::env::var_os(REQUIRED_ENV).is_none(), "{message}");
            println!("{message}");
            eprintln!("{message}");
        }
        ports
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
    /// does (only those with a smaller id), reporting each membership change and each of its
    /// connections that closes. `spec` is `GRACE_MS ID@PORT PEER_ID@PEER_PORT...`.
    fn run_node(spec: &str) {
        let mut parts = spec.split_whitespace();
        let grace = Duration::from_millis(
            parts
                .next()
                .and_then(|ms| ms.parse().ok())
                .expect("grace in ms"),
        );
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
                let closed = z
                    .info()
                    .transport_events_listener()
                    .history(false)
                    .await
                    .expect("transport events");
                tokio::spawn(async move {
                    while let Ok(event) = closed.recv_async().await {
                        if event.kind() == SampleKind::Delete {
                            eprintln!("SWARM closed {}", event.transport().zid());
                        }
                    }
                });
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
                let mut stream = Swarm::with_grace(session, from_client, grace).into_stream();
                while let Some(event) = stream.next().await {
                    match event {
                        FromSwarm::Membership(member) => {
                            let state = match member.state {
                                MemberState::Alive => "Alive".to_owned(),
                                MemberState::Resuming => "Resuming".to_owned(),
                                MemberState::Suspect(reason) => format!("Suspect({reason:?})"),
                                MemberState::Gone(reason) => format!("Gone({reason:?})"),
                            };
                            eprintln!("SWARM {state} {}", member.node);
                        }
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
