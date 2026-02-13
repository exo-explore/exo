use libp2p::{
    Multiaddr, PeerId,
    futures::StreamExt,
    gossipsub::{self, TopicHash},
    identify,
    identity::Keypair,
    mdns,
    swarm::{NetworkBehaviour, SwarmEvent, dial_opts::DialOpts},
};
use std::collections::HashMap;
use tokio::sync::mpsc;

#[derive(Debug)]
pub struct ListenError;

pub enum FromSwarm {
    PublishError(gossipsub::PublishError),
    Discovered(PeerId),
    Expired(PeerId),
    Message(PeerId, String, Vec<u8>),
}
pub enum ToSwarm {
    Message(String, Vec<u8>),
    Subscribe(String),
    Unsubscribe(String),
}

pub struct Peer {
    pub swarm: libp2p::Swarm<Behaviour>,
    to_client: mpsc::Sender<FromSwarm>,
    from_client: mpsc::Receiver<ToSwarm>,
    namespace: String,
    known_peers: HashMap<PeerId, Vec<Multiaddr>>,
}
impl Peer {
    pub fn new(
        namespace: String,
        kp: Keypair,
        to_client: mpsc::Sender<FromSwarm>,
        from_client: mpsc::Receiver<ToSwarm>,
    ) -> Result<Self, ListenError> {
        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(kp)
            .with_tokio()
            .with_quic()
            // TODO(evan) .with_bandwidth_metrics()
            .with_behaviour(|kp| Behaviour::new(namespace.clone(), kp))
            .expect("invalid swarm behaviour")
            .build();

        swarm
            .listen_on("/ip6/::/udp/0/quic-v1".parse().expect("invalid multiaddr"))
            .map_err(|_| ListenError)?;
        swarm
            .listen_on(
                "/ip4/0.0.0.0/udp/0/quic-v1"
                    .parse()
                    .expect("invalid multiaddr"),
            )
            .map_err(|_| ListenError)?;
        Ok(Self {
            swarm,
            to_client,
            from_client,
            namespace,
            known_peers: HashMap::default(),
        })
    }
    pub async fn run(&mut self) -> Result<(), ()> {
        loop {
            tokio::select! {
                event = self.swarm.next() => self.handle_event(event.ok_or(())?).await?,
                msg = self.from_client.recv() => self.handle_message(msg.ok_or(())?).await?,
            }
        }
    }
    async fn handle_message(&mut self, message: ToSwarm) -> Result<(), ()> {
        match message {
            ToSwarm::Message(topic, data) => {
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(TopicHash::from_raw(topic), data)
                {
                    self.to_client
                        .send(FromSwarm::PublishError(e))
                        .await
                        .map_err(|_| ())?;
                }
            }
            ToSwarm::Subscribe(topic) => {
                match self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .subscribe(&gossipsub::IdentTopic::new(topic))
                {
                    Ok(_) => {}
                    Err(gossipsub::SubscriptionError::NotAllowed) => {
                        unreachable!("subscription filter hit")
                    }
                    Err(gossipsub::SubscriptionError::PublishError(e)) => self
                        .to_client
                        .send(FromSwarm::PublishError(e))
                        .await
                        .map_err(|_| ())?,
                }
            }
            ToSwarm::Unsubscribe(topic) => {
                self.swarm
                    .behaviour_mut()
                    .gossipsub
                    .unsubscribe(&gossipsub::IdentTopic::new(topic));
            }
        }
        Ok(())
    }
    async fn handle_event(&mut self, event: SwarmEvent<BehaviourEvent>) -> Result<(), ()> {
        let SwarmEvent::Behaviour(event) = event else {
            return Ok(());
        };
        match event {
            BehaviourEvent::Gossipsub(gossipsub::Event::Message { message, .. }) => {
                if let Some(source) = message.source {
                    self.to_client
                        .send(FromSwarm::Message(
                            source,
                            message.topic.into_string(),
                            message.data,
                        ))
                        .await
                        .map_err(|_| ())?;
                }
            }
            BehaviourEvent::Identify(identify::Event::Received { peer_id, info, .. }) => {
                log::debug!(
                    "identify from {peer_id}: protocol_version='{}' agent_version='{}' (local namespace='{}')",
                    info.protocol_version,
                    info.agent_version,
                    self.namespace
                );
                if info.protocol_version == self.namespace {
                    self.passed_namespace(peer_id);
                    self.to_client
                        .send(FromSwarm::Discovered(peer_id))
                        .await
                        .map_err(|_| ())?;
                } else {
                    self.failed_namespace(peer_id);
                }
            }
            BehaviourEvent::Mdns(mdns::Event::Discovered(v)) => {
                for (peer_id, addr) in v {
                    self.known_peers.entry(peer_id).or_default().push(addr);
                }
                for (peer_id, addrs) in &self.known_peers {
                    // dialopts handles rate limiting, we should check errors if we want to blacklist earlier
                    let _ = self
                        .swarm
                        .dial(DialOpts::peer_id(*peer_id).addresses(addrs.clone()).build());
                }
            }
            BehaviourEvent::Mdns(mdns::Event::Expired(v)) => {
                for (peer_id, addr) in v {
                    let addrs = self.known_peers.entry(peer_id).or_default();
                    addrs.retain(|a| *a != addr);
                    if addrs.is_empty() {
                        self.known_peers.remove(&peer_id);
                        self.swarm
                            .behaviour_mut()
                            .gossipsub
                            .remove_explicit_peer(&peer_id);
                        self.to_client
                            .send(FromSwarm::Expired(peer_id))
                            .await
                            .map_err(|_| ())?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn passed_namespace(&mut self, peer_id: PeerId) {
        self.swarm
            .behaviour_mut()
            .gossipsub
            .remove_blacklisted_peer(&peer_id);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .add_explicit_peer(&peer_id);
    }
    fn failed_namespace(&mut self, peer_id: PeerId) {
        self.swarm
            .behaviour_mut()
            .gossipsub
            .blacklist_peer(&peer_id);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .remove_explicit_peer(&peer_id);
    }
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    identify: identify::Behaviour,
}

impl Behaviour {
    fn new(namespace: String, kp: &Keypair) -> Self {
        let mdns = mdns::Behaviour::new(mdns::Config::default(), kp.public().to_peer_id())
            .expect("mdns behaviour failed to build");

        let identify =
            identify::Behaviour::new(identify::Config::new_with_signed_peer_record(namespace, kp));

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(kp.clone()),
            gossipsub::ConfigBuilder::default()
                .max_transmit_size(1024 * 1024)
                .validation_mode(gossipsub::ValidationMode::Strict)
                .build()
                .expect("invalid gossipsub configuration"),
        )
        .expect("gossipsub behaviour failed ot build");

        Self {
            gossipsub,
            mdns,
            identify,
        }
    }
}

// TODO: more tests
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, timeout};

    fn make_peer(namespace: &str) -> (Peer, mpsc::Receiver<FromSwarm>, mpsc::Sender<ToSwarm>) {
        let kp = Keypair::generate_ed25519();

        let (to_client_tx, to_client_rx) = mpsc::channel(64);
        let (to_peer_tx, to_peer_rx) = mpsc::channel(64);

        let peer = Peer::new(namespace.to_string(), kp, to_client_tx, to_peer_rx)
            .expect("Peer::new should succeed in tests");

        (peer, to_client_rx, to_peer_tx)
    }

    async fn next_listen_addr(peer: &mut Peer) -> Multiaddr {
        loop {
            match peer.swarm.next().await {
                Some(SwarmEvent::NewListenAddr { address, .. }) => return address,
                Some(_) => {}
                None => panic!("swarm stream ended unexpectedly"),
            }
        }
    }

    #[tokio::test]
    async fn subscribe_and_unsubscribe_do_not_error() {
        let (mut peer, mut events_rx, commands_tx) = make_peer("ns-test");

        // Drive the swarm just enough to get at least one listen address event,
        // so the background run loop has something initialized.
        let _addr = next_listen_addr(&mut peer).await;

        // Run the peer loop in the background.
        let handle = tokio::spawn(async move {
            let _ = peer.run().await;
        });

        commands_tx
            .send(ToSwarm::Subscribe("topic-a".to_string()))
            .await
            .unwrap();

        commands_tx
            .send(ToSwarm::Unsubscribe("topic-a".to_string()))
            .await
            .unwrap();

        // We don't *require* any FromSwarm events here; this is mainly a
        // smoke test that the message-handling path doesn't panic/hang.
        // Still, poll briefly to ensure the task is alive.
        let _ = timeout(Duration::from_millis(200), events_rx.recv()).await;

        // Shut down: dropping the command sender closes the channel, causing run() to return Err.
        drop(commands_tx);
        let _ = handle.await;
    }
}
