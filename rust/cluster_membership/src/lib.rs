use libp2p::{
    Multiaddr, PeerId, Swarm, SwarmBuilder,
    futures::StreamExt,
    gossipsub::{self, PublishError, Sha256Topic, TopicHash},
    identify,
    identity::{Keypair, ed25519},
    mdns,
    swarm::{NetworkBehaviour, SwarmEvent, dial_opts::DialOpts},
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::{select, sync::mpsc};

const DEFAULT_BUFFER_SIZE: usize = 10;
const MDNS_IGNORE_DURATION_SECS: u64 = 30;

impl Peer {
    pub fn new(
        identity: ed25519::SecretKey,
        namespace: String,
    ) -> anyhow::Result<(
        Self,
        mpsc::Sender<(TopicHash, Vec<u8>)>,
        mpsc::Receiver<Result<(TopicHash, PeerId, Vec<u8>), PublishError>>,
    )> {
        let mut id_bytes = identity.as_ref().to_vec();

        let mut swarm =
            SwarmBuilder::with_existing_identity(Keypair::ed25519_from_bytes(&mut id_bytes)?)
                .with_tokio()
                .with_quic()
                // TODO(evan): .with_bandwidth_metrics();
                .with_behaviour(|kp| Behaviour::new(kp, namespace.clone()))?
                .build();

        swarm.listen_on("/ip6/::/udp/0/quic-v1".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        let (to_swarm, from_client) = mpsc::channel(DEFAULT_BUFFER_SIZE);
        let (to_client, from_swarm) = mpsc::channel(DEFAULT_BUFFER_SIZE);
        Ok((
            Self {
                swarm,
                namespace,
                denied: HashMap::new(),
                from_client,
                to_client,
            },
            to_swarm,
            from_swarm,
        ))
    }

    pub fn subscribe(&mut self, topic: String) -> TopicHash {
        let topic = Sha256Topic::new(topic);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .expect("topic filtered");
        topic.hash()
    }

    pub async fn run(&mut self) {
        loop {
            select! {
                ev = self.swarm.select_next_some() => {
                    let Ok(()) = self.handle_swarm_event(ev).await else {
                        return
                    };
                },
                Some(msg) = self.from_client.recv() => {
                    if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(msg.0, msg.1) {
                        let Ok(()) = self.to_client.send(Err(e)).await else {
                            return
                        };
                    }
                },
            }
        }
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<BehaviourEvent>) -> Result<(), ()> {
        let SwarmEvent::Behaviour(event) = event else {
            if let SwarmEvent::NewListenAddr {
                listener_id: _,
                address,
            } = event
            {
                log::info!("new listen address {address}")
            }
            return Ok(());
        };
        match event {
            BehaviourEvent::Mdns(mdns_event) => match mdns_event {
                mdns::Event::Discovered(vec) => {
                    // Dial everyone
                    let mut addrs = HashMap::<PeerId, Vec<Multiaddr>>::new();
                    vec.into_iter()
                        .filter(|(peer_id, _)| {
                            self.denied.get(peer_id).is_none_or(|t| {
                                t.elapsed() > Duration::from_secs(MDNS_IGNORE_DURATION_SECS)
                            })
                        })
                        .for_each(|(peer_id, addr)| addrs.entry(peer_id).or_default().push(addr));
                    addrs.into_iter().for_each(|(peer_id, addrs)| {
                        let _ = self
                            .swarm
                            .dial(DialOpts::peer_id(peer_id).addresses(addrs).build());
                    });
                }
                mdns::Event::Expired(vec) => {
                    vec.iter().for_each(|(peer_id, _)| {
                        log::debug!("{peer_id} no longer reachable on mDNS");
                        self.swarm
                            .behaviour_mut()
                            .gossipsub
                            .remove_explicit_peer(peer_id);
                    });
                }
            },
            BehaviourEvent::Identify(identify::Event::Received {
                connection_id: _,
                peer_id,
                info,
            }) => {
                if info
                    .protocols
                    .iter()
                    .any(|p| p.as_ref().contains(&self.namespace))
                {
                    self.passed_namespace(peer_id);
                } else {
                    self.failed_namespace(peer_id);
                }
            }
            BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source: _,
                message_id: _,
                message:
                    gossipsub::Message {
                        topic,
                        data,
                        source: Some(source_peer),
                        ..
                    },
            }) => {
                let Ok(()) = self.to_client.send(Ok((topic, source_peer, data))).await else {
                    return Err(());
                };
            }
            _ => {}
        }
        Ok(())
    }

    fn passed_namespace(&mut self, peer: PeerId) {
        log::info!("new peer {peer:?}");
        self.denied.remove(&peer);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .remove_blacklisted_peer(&peer);
        self.swarm
            .behaviour_mut()
            .gossipsub
            .add_explicit_peer(&peer);
    }

    fn failed_namespace(&mut self, peer: PeerId) {
        log::debug!("{peer} failed handshake");
        self.denied.insert(peer, Instant::now());
        self.swarm.behaviour_mut().gossipsub.blacklist_peer(&peer);
        // we don't care if disconnect fails
        let _ = self.swarm.disconnect_peer_id(peer);
    }
}

pub struct Peer {
    pub swarm: Swarm<Behaviour>,
    denied: HashMap<PeerId, Instant>,
    namespace: String,
    to_client: mpsc::Sender<Result<(TopicHash, PeerId, Vec<u8>), PublishError>>,
    from_client: mpsc::Receiver<(TopicHash, Vec<u8>)>,
}

#[test]
fn foo() {
    fn bar<T: Send>(t: T) {}
    let p: Peer = unimplemented!();
    bar(p);
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    mdns: mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
}

impl Behaviour {
    fn new(kp: &Keypair, namespace: String) -> Self {
        let mdns = mdns::tokio::Behaviour::new(Default::default(), kp.public().to_peer_id())
            .expect("implementation is infallible");
        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(kp.clone()),
            gossipsub::ConfigBuilder::default()
                .max_transmit_size(1024 * 1024)
                .protocol_id_prefix(format!("/exo/gossip/{namespace}/v1"))
                .build()
                .expect("fixed gossipsub config should always build"),
        )
        .expect("fixed gossipsub init should always build");

        let identify = identify::Behaviour::new(
            identify::Config::new_with_signed_peer_record(format!("/exo/identity/v1"), kp)
                .with_push_listen_addr_updates(true),
        );

        Behaviour {
            mdns,
            gossipsub,
            identify,
        }
    }
}
