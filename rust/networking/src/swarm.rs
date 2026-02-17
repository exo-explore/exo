use crate::alias;
use crate::discovery;
use crate::swarm::transport::tcp_transport;
use behaviour::{Behaviour, BehaviourEvent};
use futures_lite::StreamExt;
use libp2p::{PeerId, SwarmBuilder, gossipsub, identity, swarm::SwarmEvent};
use tokio::sync::mpsc;

pub struct Swarm {
    swarm: libp2p::Swarm<Behaviour>,
    from_client: mpsc::Receiver<ToSwarm>,
    to_client: mpsc::Sender<FromSwarm>,
}

#[derive(Debug)]
pub enum FromSwarm {
    PublishError(gossipsub::PublishError),
    Discovered(PeerId),
    Expired(PeerId),
    Message(PeerId, String, Vec<u8>),
}
#[derive(Debug)]
pub enum ToSwarm {
    Message(String, Vec<u8>),
    Subscribe(String),
    Unsubscribe(String),
}

/// The current version of the network: this prevents devices running different versions of the
/// software from interacting with each other.
///
/// TODO: right now this is a hardcoded constant; figure out what the versioning semantics should
///       even be, and how to inject the right version into this config/initialization. E.g. should
///       this be passed in as a parameter? What about rapidly changing versions in debug builds?
///       this is all VERY very hard to figure out and needs to be mulled over as a team.
pub const NETWORK_VERSION: &[u8] = b"v0.0.1";
pub const OVERRIDE_VERSION_ENV_VAR: &str = "EXO_LIBP2P_NAMESPACE";

impl Swarm {
    /// Create and configure a swarm which listens to all ports on OS
    pub fn new(
        keypair: identity::Keypair,
        from_client: mpsc::Receiver<ToSwarm>,
        to_client: mpsc::Sender<FromSwarm>,
    ) -> alias::AnyResult<Swarm> {
        let mut swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_other_transport(tcp_transport)?
            .with_behaviour(Behaviour::new)?
            .build();

        // Listen on all interfaces and whatever port the OS assigns
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
        Ok(Self {
            swarm,
            from_client,
            to_client,
        })
    }
    pub async fn run(&mut self) {
        log::info!("RUST: networking task started");

        loop {
            tokio::select! {
                message = self.from_client.recv() => {
                    // handle closed channel
                    let Some(message) = message else {
                        log::info!("RUST: channel closed");
                        break;
                    };

                    // dispatch incoming messages
                    match message {
                        ToSwarm::Subscribe(topic) => {
                            // try to subscribe
                            match self.swarm.behaviour_mut().gossipsub.subscribe(&gossipsub::IdentTopic::new(topic.clone())) {
                                    Err(e) => {
                                        let gossipsub::SubscriptionError::PublishError(e) = e else {
                                            unreachable!("topic filter used")
                                        };
                                        let Ok(()) = self.to_client.send(FromSwarm::PublishError(e)).await else {
                                            log::warn!("RUST: client connection closed");
                                            break
                                        };
                                    },
                                    Ok(false) => log::warn!("RUST: tried to subscribe to topic twice"),
                                    Ok(true) => {},
                                }
                        }
                        ToSwarm::Unsubscribe(topic) => {
                            // try to subscribe
                            if !self.swarm.behaviour_mut().gossipsub.unsubscribe(&gossipsub::IdentTopic::new(topic)) {
                                log::warn!("RUST: tried to unsubscribe from topic twice");
                            }
                        }
                        ToSwarm::Message( topic, data ) => {
                            // try to publish the data -> catch NoPeersSubscribedToTopic error & convert to correct exception
                            match self.swarm.behaviour_mut().gossipsub.publish(
                                gossipsub::IdentTopic::new(topic), data
                            ) {
                                Ok(_) => {},
                                Err(e) => {
                                    let Ok(()) = self.to_client.send(FromSwarm::PublishError(e)).await else {
                                        log::warn!("RUST: client connection closed");
                                        break
                                    };
                                },
                            }
                        }
                    }
                }

                // architectural solution to this problem:
                // create keep_alive behavior who's job it is to dial peers discovered by mDNS (and drop when expired)
                //   -> it will emmit TRUE connected/disconnected events consumable elsewhere
                //
                // gossipsub will feed off-of dial attempts created by networking, and that will bootstrap its' peers list
                // then for actual communication it will dial those peers if need-be
                swarm_event = self.swarm.next() => {
                    let Some(swarm_event) = swarm_event else {
                        log::warn!("RUST: swarm closed communication");
                        break
                    };
                    let SwarmEvent::Behaviour(behaviour_event) = swarm_event else {
                        continue
                    };
                    match behaviour_event {
                        BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            message: gossipsub::Message {
                                source,
                                topic,
                                data,
                                ..
                            },
                            ..
                        }) => {
                            let Some(peer_id) = source else {
                                log::warn!("RUST: ignoring message with unknown source on {topic}");
                                continue;
                            };
                            // send incoming message to channel (or exit if connection closed)
                            if let Err(e) = self.to_client.send(FromSwarm::Message(peer_id, topic.into_string(), data)).await {
                                log::warn!("RUST: could not send incoming gossipsub message since channel already closed: {e}");
                                break
                            };
                        },
                        BehaviourEvent::Discovery(discovery::Event::ConnectionEstablished { peer_id, .. }) => {
                            // send connection event to channel (or exit if connection closed)
                            if let Err(_) = self.to_client.send(FromSwarm::Discovered(peer_id)).await {
                                log::warn!("RUST: swarm closed communication");
                            };
                        },
                        BehaviourEvent::Discovery(discovery::Event::ConnectionClosed { peer_id, .. }) => {
                            // send connection event to channel (or exit if connection closed)
                            if let Err(_) = self.to_client.send(FromSwarm::Expired(peer_id)).await {
                                log::warn!("RUST: swarm closed communication");
                            };
                        },
                        e => {
                            log::debug!("RUST: other event {e:?}");
                        }
                    }
                }
            }
        }

        log::info!("RUST: networking task stopped");
    }
}

mod transport {
    use crate::alias;
    use crate::swarm::{NETWORK_VERSION, OVERRIDE_VERSION_ENV_VAR};
    use futures_lite::{AsyncRead, AsyncWrite};
    use keccak_const::Sha3_256;
    use libp2p::core::muxing;
    use libp2p::core::transport::Boxed;
    use libp2p::pnet::{PnetError, PnetOutput};
    use libp2p::{PeerId, Transport, identity, noise, pnet, yamux};
    use std::{env, sync::LazyLock};

    /// Key used for networking's private network; parametrized on the [`NETWORK_VERSION`].
    /// See [`pnet_upgrade`] for more.
    static PNET_PRESHARED_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
        let builder = Sha3_256::new().update(b"exo_discovery_network");

        if let Ok(var) = env::var(OVERRIDE_VERSION_ENV_VAR) {
            let bytes = var.into_bytes();
            builder.update(&bytes)
        } else {
            builder.update(NETWORK_VERSION)
        }
        .finalize()
    });

    /// Make the Swarm run on a private network, as to not clash with public libp2p nodes and
    /// also different-versioned instances of this same network.
    /// This is implemented as an additional "upgrade" ontop of existing [`libp2p::Transport`] layers.
    async fn pnet_upgrade<TSocket>(
        socket: TSocket,
        _: impl Sized,
    ) -> Result<PnetOutput<TSocket>, PnetError>
    where
        TSocket: AsyncRead + AsyncWrite + Send + Unpin + 'static,
    {
        use pnet::{PnetConfig, PreSharedKey};
        PnetConfig::new(PreSharedKey::new(*PNET_PRESHARED_KEY))
            .handshake(socket)
            .await
    }

    /// TCP/IP transport layer configuration.
    pub fn tcp_transport(
        keypair: &identity::Keypair,
    ) -> alias::AnyResult<Boxed<(PeerId, muxing::StreamMuxerBox)>> {
        use libp2p::{
            core::upgrade::Version,
            tcp::{Config, tokio},
        };

        // `TCP_NODELAY` enabled => avoid latency
        let tcp_config = Config::default().nodelay(true);

        // V1 + lazy flushing => 0-RTT negotiation
        let upgrade_version = Version::V1Lazy;

        // Noise is faster than TLS + we don't care much for security
        let noise_config = noise::Config::new(keypair)?;

        // Use default Yamux config for multiplexing
        let yamux_config = yamux::Config::default();

        // Create new Tokio-driven TCP/IP transport layer
        let base_transport = tokio::Transport::new(tcp_config)
            .and_then(pnet_upgrade)
            .upgrade(upgrade_version)
            .authenticate(noise_config)
            .multiplex(yamux_config);

        // Return boxed transport (to flatten complex type)
        Ok(base_transport.boxed())
    }
}

mod behaviour {
    use crate::{alias, discovery};
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{gossipsub, identity};

    /// Behavior of the Swarm which composes all desired behaviors:
    /// Right now its just [`discovery::Behaviour`] and [`gossipsub::Behaviour`].
    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        pub discovery: discovery::Behaviour,
        pub gossipsub: gossipsub::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> alias::AnyResult<Self> {
            Ok(Self {
                discovery: discovery::Behaviour::new(keypair)?,
                gossipsub: gossipsub_behaviour(keypair),
            })
        }
    }

    fn gossipsub_behaviour(keypair: &identity::Keypair) -> gossipsub::Behaviour {
        use gossipsub::{ConfigBuilder, MessageAuthenticity, ValidationMode};

        // build a gossipsub network behaviour
        //  => signed message authenticity + strict validation mode means the message-ID is
        //     automatically provided by gossipsub w/out needing to provide custom message-ID function
        gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(keypair.clone()),
            ConfigBuilder::default()
                .max_transmit_size(1024 * 1024)
                .validation_mode(ValidationMode::Strict)
                .build()
                .expect("the configuration should always be valid"),
        )
        .expect("creating gossipsub behavior should always work")
    }
}
