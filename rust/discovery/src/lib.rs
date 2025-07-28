//! TODO: crate documentation
//!
//! this is here as a placeholder documentation
//!
//!

// enable Rust-unstable features for convenience
#![feature(trait_alias)]
// #![feature(stmt_expr_attributes)]
// #![feature(unboxed_closures)]
// #![feature(assert_matches)]
// #![feature(async_fn_in_dyn_trait)]
// #![feature(async_for_loop)]
// #![feature(auto_traits)]
// #![feature(negative_impls)]

use crate::behaviour::{discovery_behaviour, DiscoveryBehaviour};
use crate::transport::discovery_transport;
use libp2p::{identity, Swarm, SwarmBuilder};

pub mod behaviour;
pub mod transport;

/// Namespace for all the type/trait aliases used by this crate.
pub(crate) mod alias {
    use std::error::Error;

    pub type AnyError = Box<dyn Error + Send + Sync + 'static>;
    pub type AnyResult<T> = Result<T, AnyError>;
}

/// Namespace for crate-wide extension traits/methods
pub(crate) mod ext {}

pub(crate) mod private {
    /// Sealed traits support
    pub trait Sealed {}
    impl<T: ?Sized> Sealed for T {}
}

/// Create and configure a swarm, and start listening to all ports/OS.
#[inline]
pub fn discovery_swarm(keypair: identity::Keypair) -> alias::AnyResult<Swarm<DiscoveryBehaviour>> {
    let peer_id = keypair.public().to_peer_id();
    log::info!("RUST: Creating discovery swarm with peer_id: {}", peer_id);
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_other_transport(discovery_transport)?
        .with_behaviour(discovery_behaviour)?
        .build();

    // Listen on all interfaces and whatever port the OS assigns
    // swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?; // TODO: make this
    let listen_addr = "/ip4/0.0.0.0/tcp/0".parse()?;
    log::info!("RUST: Attempting to listen on: {}", listen_addr);
    swarm.listen_on(listen_addr)?;

    Ok(swarm)
}

// TODO:  - ensure that all changes to connections means a Disconnect/Reconnect event fired, i.e. if it switched IPs slighty or something
//        - ensure that all links are unique, i.e. each connection has some kind of uniquely identifiable hash/multiaddress/whatever => temporally unique???
//        - need pnet config, so that forwarder & discovery don't interfere with each-other
//        - discovery network needs persistence, so swarm created from existing identity (passed as arg)
//        - connect/disconnect events etc. should be handled with callbacks
//        - DON'T need gossipsub JUST yet, only mDNS for discovery => potentially use something else instead of gossipsub

#[cfg(test)]
mod tests {
    use crate::alias::AnyResult;
    use crate::behaviour::DiscoveryBehaviourEvent;
    use crate::discovery_swarm;
    use futures::stream::StreamExt as _;
    use libp2p::{gossipsub, identity, mdns, swarm::SwarmEvent};
    use std::hash::Hash;
    use tokio::{io, io::AsyncBufReadExt as _, select};
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::util::SubscriberInitExt as _;
    use tracing_subscriber::EnvFilter;

    #[tokio::test]
    async fn chatroom_test() -> AnyResult<()> {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::DEBUG.into()))
            .try_init();

        // Configure swarm
        let mut swarm = discovery_swarm(identity::Keypair::generate_ed25519())?;

        // Create a Gossipsub topic & subscribe
        let topic = gossipsub::IdentTopic::new("test-net");
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

        // Read full lines from stdin
        let mut stdin = io::BufReader::new(io::stdin()).lines();
        println!(
            "Enter messages via STDIN and they will be sent to connected peers using Gossipsub"
        );

        // Kick it off
        loop {
            select! {
                Ok(Some(line)) = stdin.next_line() => {
                    if let Err(e) = swarm
                        .behaviour_mut().gossipsub
                        .publish(topic.clone(), line.as_bytes()) {
                        println!("Publish error: {e:?}");
                    }
                }
                event = swarm.select_next_some() => match event {
                    SwarmEvent::Behaviour(DiscoveryBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, multiaddr) in list {
                            println!("mDNS discovered a new peer: {peer_id} on {multiaddr}");
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                        }
                    },
                    SwarmEvent::Behaviour(DiscoveryBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, multiaddr) in list {
                            println!("mDNS discover peer has expired: {peer_id} on {multiaddr}");
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                        }
                    },
                    SwarmEvent::Behaviour(DiscoveryBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id,
                        message_id: id,
                        message,
                    })) => println!(
                            "\n\nGot message: '{}' with id: {id} from peer: {peer_id}\n\n",
                            String::from_utf8_lossy(&message.data),
                        ),
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("Local node is listening on {address}");
                    }
                    e => {
                        println!("Other event {e:?}");
                    }
                }
            }
        }
    }
}
