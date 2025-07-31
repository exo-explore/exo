use crate::alias::AnyResult;
use libp2p::core::Endpoint;
use libp2p::core::transport::PortUse;
use libp2p::swarm::derive_prelude::Either;
use libp2p::swarm::{
    ConnectionDenied, ConnectionHandler, ConnectionHandlerSelect, ConnectionId, FromSwarm,
    NetworkBehaviour, THandler, THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use libp2p::{Multiaddr, PeerId, gossipsub, identity, mdns};
use std::fmt;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

/// Custom network behavior for `discovery` network; it combines [`mdns::tokio::Behaviour`] for
/// the actual mDNS discovery, and [`gossipsub::Behaviour`] for PubSub functionality.
#[derive(NetworkBehaviour)]
pub struct DiscoveryBehaviour {
    pub mdns: mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
}

// #[doc = "`NetworkBehaviour::ToSwarm` produced by DiscoveryBehaviour."]
// pub enum DiscoveryBehaviourEvent {
//     Mdns(<mdns::tokio::Behaviour as NetworkBehaviour>::ToSwarm),
//     Gossipsub(<gossipsub::Behaviour as NetworkBehaviour>::ToSwarm),
// }
// impl Debug for DiscoveryBehaviourEvent
// where
//     <mdns::tokio::Behaviour as NetworkBehaviour>::ToSwarm: Debug,
//     <gossipsub::Behaviour as NetworkBehaviour>::ToSwarm: Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
//         match &self {
//             DiscoveryBehaviourEvent::Mdns(event) => {
//                 f.write_fmt(format_args!("{}: {:?}", "DiscoveryBehaviourEvent", event))
//             }
//             DiscoveryBehaviourEvent::Gossipsub(event) => {
//                 f.write_fmt(format_args!("{}: {:?}", "DiscoveryBehaviourEvent", event))
//             }
//         }
//     }
// }
// impl NetworkBehaviour for DiscoveryBehaviour
// where
//     mdns::tokio::Behaviour: NetworkBehaviour,
//     gossipsub::Behaviour: NetworkBehaviour,
// {
//     type ConnectionHandler =
//         ConnectionHandlerSelect<THandler<mdns::tokio::Behaviour>, THandler<gossipsub::Behaviour>>;
//     type ToSwarm = DiscoveryBehaviourEvent;
//     #[allow(clippy::needless_question_mark)]
//     fn handle_pending_inbound_connection(
//         &mut self,
//         connection_id: ConnectionId,
//         local_addr: &Multiaddr,
//         remote_addr: &Multiaddr,
//     ) -> Result<(), ConnectionDenied> {
//         NetworkBehaviour::handle_pending_inbound_connection(
//             &mut self.mdns,
//             connection_id,
//             local_addr,
//             remote_addr,
//         )?;
//         NetworkBehaviour::handle_pending_inbound_connection(
//             &mut self.gossipsub,
//             connection_id,
//             local_addr,
//             remote_addr,
//         )?;
//         Ok(())
//     }
//     #[allow(clippy::needless_question_mark)]
//     fn handle_established_inbound_connection(
//         &mut self,
//         connection_id: ConnectionId,
//         peer: PeerId,
//         local_addr: &Multiaddr,
//         remote_addr: &Multiaddr,
//     ) -> Result<THandler<Self>, ConnectionDenied> {
//         Ok(ConnectionHandler::select(
//             self.mdns.handle_established_inbound_connection(
//                 connection_id,
//                 peer,
//                 local_addr,
//                 remote_addr,
//             )?,
//             self.gossipsub.handle_established_inbound_connection(
//                 connection_id,
//                 peer,
//                 local_addr,
//                 remote_addr,
//             )?,
//         ))
//     }
//     #[allow(clippy::needless_question_mark)]
//     fn handle_pending_outbound_connection(
//         &mut self,
//         connection_id: ConnectionId,
//         maybe_peer: Option<PeerId>,
//         addresses: &[Multiaddr],
//         effective_role: Endpoint,
//     ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
//         let mut combined_addresses = Vec::new();
//         combined_addresses.extend(NetworkBehaviour::handle_pending_outbound_connection(
//             &mut self.mdns,
//             connection_id,
//             maybe_peer,
//             addresses,
//             effective_role,
//         )?);
//         combined_addresses.extend(NetworkBehaviour::handle_pending_outbound_connection(
//             &mut self.gossipsub,
//             connection_id,
//             maybe_peer,
//             addresses,
//             effective_role,
//         )?);
//         Ok(combined_addresses)
//     }
//     #[allow(clippy::needless_question_mark)]
//     fn handle_established_outbound_connection(
//         &mut self,
//         connection_id: ConnectionId,
//         peer: PeerId,
//         addr: &Multiaddr,
//         role_override: Endpoint,
//         port_use: PortUse,
//     ) -> Result<THandler<Self>, ConnectionDenied> {
//         Ok(ConnectionHandler::select(
//             self.mdns.handle_established_outbound_connection(
//                 connection_id,
//                 peer,
//                 addr,
//                 role_override,
//                 port_use,
//             )?,
//             self.gossipsub.handle_established_outbound_connection(
//                 connection_id,
//                 peer,
//                 addr,
//                 role_override,
//                 port_use,
//             )?,
//         ))
//     }
//     fn on_swarm_event(&mut self, event: FromSwarm) {
//         self.mdns.on_swarm_event(event);
//         self.gossipsub.on_swarm_event(event);
//     }
//     fn on_connection_handler_event(
//         &mut self,
//         peer_id: PeerId,
//         connection_id: ConnectionId,
//         event: THandlerOutEvent<Self>,
//     ) {
//         match event {
//             Either::Left(ev) => NetworkBehaviour::on_connection_handler_event(
//                 &mut self.mdns,
//                 peer_id,
//                 connection_id,
//                 ev,
//             ),
//             Either::Right(ev) => NetworkBehaviour::on_connection_handler_event(
//                 &mut self.gossipsub,
//                 peer_id,
//                 connection_id,
//                 ev,
//             ),
//         }
//     }
//     fn poll(
//         &mut self,
//         cx: &mut std::task::Context,
//     ) -> std::task::Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
//         match NetworkBehaviour::poll(&mut self.mdns, cx) {
//             std::task::Poll::Ready(e) => {
//                 return std::task::Poll::Ready(
//                     e.map_out(DiscoveryBehaviourEvent::Mdns)
//                         .map_in(|event| Either::Left(event)),
//                 );
//             }
//             std::task::Poll::Pending => {}
//         }
//         match NetworkBehaviour::poll(&mut self.gossipsub, cx) {
//             std::task::Poll::Ready(e) => {
//                 return std::task::Poll::Ready(
//                     e.map_out(DiscoveryBehaviourEvent::Gossipsub)
//                         .map_in(|event| Either::Right(event)),
//                 );
//             }
//             std::task::Poll::Pending => {}
//         }
//         std::task::Poll::Pending
//     }
// }

fn mdns_behaviour(keypair: &identity::Keypair) -> AnyResult<mdns::tokio::Behaviour> {
    use mdns::{Config, tokio};

    // mDNS config => enable IPv6
    let mdns_config = Config {
        enable_ipv6: true,
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
