use crate::alias;
use crate::swarm::transport::tcp_transport;
pub use behaviour::{Behaviour, BehaviourEvent};
use libp2p::{SwarmBuilder, identity};

pub type Swarm = libp2p::Swarm<Behaviour>;

/// The current version of the network: this prevents devices running different versions of the
/// software from interacting with each other.
///
/// TODO: right now this is a hardcoded constant; figure out what the versioning semantics should
///       even be, and how to inject the right version into this config/initialization. E.g. should
///       this be passed in as a parameter? What about rapidly changing versions in debug builds?
///       this is all VERY very hard to figure out and needs to be mulled over as a team.
pub const NETWORK_VERSION: &[u8] = b"v0.0.1";
pub const OVERRIDE_VERSION_ENV_VAR: &str = "EXO_LIBP2P_NAMESPACE";

/// Create and configure a swarm which listens to all ports on OS
pub fn create_swarm(keypair: identity::Keypair) -> alias::AnyResult<Swarm> {
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_other_transport(tcp_transport)?
        .with_behaviour(Behaviour::new)?
        .build();

    // Listen on all interfaces and whatever port the OS assigns
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
    Ok(swarm)
}

mod transport {
    use crate::alias;
    use crate::swarm::{NETWORK_VERSION, OVERRIDE_VERSION_ENV_VAR};
    use futures::{AsyncRead, AsyncWrite};
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
    use std::time::Duration;

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
                .publish_queue_duration(Duration::from_secs(15))
                .max_transmit_size(1024 * 1024)
                .validation_mode(ValidationMode::Strict)
                .build()
                .expect("the configuration should always be valid"),
        )
        .expect("creating gossipsub behavior should always work")
    }
}
