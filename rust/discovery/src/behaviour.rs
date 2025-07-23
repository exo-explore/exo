use crate::alias::AnyResult;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{gossipsub, identity, mdns};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

/// Custom network behavior for `discovery` network; it combines [`mdns::tokio::Behaviour`] for
/// the actual mDNS discovery, and [`gossipsub::Behaviour`] for PubSub functionality.
#[derive(NetworkBehaviour)]
pub struct DiscoveryBehaviour {
    pub mdns: mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
}

fn mdns_behaviour(keypair: &identity::Keypair) -> AnyResult<mdns::tokio::Behaviour> {
    use mdns::{tokio, Config};

    // mDNS config => enable IPv6
    let mdns_config = Config {
        // enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
        ..Default::default()
    };

    let mdns_behaviour = tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id());
    Ok(mdns_behaviour?)
}

fn gossipsub_behaviour(keypair: &identity::Keypair) -> AnyResult<gossipsub::Behaviour> {
    use gossipsub::ConfigBuilder;

    // To content-address message, we can take the hash of message and use it as an ID.
    let message_id_fn = |message: &gossipsub::Message| {
        let mut s = DefaultHasher::new();
        message.data.hash(&mut s);
        gossipsub::MessageId::from(s.finish().to_string())
    };

    let gossipsub_config = ConfigBuilder::default()
        // .mesh_n_low(1
        .mesh_n(1) // this is for debugging!!! change to 6
        // .mesh_n_for_topic(1, topic.hash()) // this is for debugging!!! change to 6
        // .mesh_n_high(1)
        .heartbeat_interval(Duration::from_secs(10)) // This is set to aid debugging by not cluttering the log space
        .validation_mode(gossipsub::ValidationMode::None) // This sets the kind of message validation. Skip signing for speed.
        .message_id_fn(message_id_fn) // content-address messages. No two messages of the same content will be propagated.
        .build()?; // Temporary hack because `build` does not return a proper `std::error::Error`.

    // build a gossipsub network behaviour
    let gossipsub_behavior = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
    )?;
    Ok(gossipsub_behavior)
}

pub fn discovery_behaviour(keypair: &identity::Keypair) -> AnyResult<DiscoveryBehaviour> {
    Ok(DiscoveryBehaviour {
        gossipsub: gossipsub_behaviour(keypair)?,
        mdns: mdns_behaviour(keypair)?,
    })
}
