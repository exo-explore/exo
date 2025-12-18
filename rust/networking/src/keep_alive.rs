use delegate::delegate;
use libp2p::swarm::handler::ConnectionEvent;
use libp2p::swarm::{ConnectionHandlerEvent, SubstreamProtocol, dummy, handler};
use std::task::{Context, Poll};

/// An implementation of [`ConnectionHandler`] that doesn't handle any protocols, but it keeps
/// the connection alive.
#[derive(Clone)]
#[repr(transparent)]
pub struct ConnectionHandler(dummy::ConnectionHandler);

impl ConnectionHandler {
    pub fn new() -> Self {
        ConnectionHandler(dummy::ConnectionHandler)
    }
}

impl handler::ConnectionHandler for ConnectionHandler {
    // delegate types and implementation mostly to dummy handler
    type FromBehaviour = <dummy::ConnectionHandler as handler::ConnectionHandler>::FromBehaviour;
    type ToBehaviour = <dummy::ConnectionHandler as handler::ConnectionHandler>::ToBehaviour;
    type InboundProtocol =
        <dummy::ConnectionHandler as handler::ConnectionHandler>::InboundProtocol;
    type OutboundProtocol =
        <dummy::ConnectionHandler as handler::ConnectionHandler>::OutboundProtocol;
    type InboundOpenInfo =
        <dummy::ConnectionHandler as handler::ConnectionHandler>::InboundOpenInfo;
    type OutboundOpenInfo =
        <dummy::ConnectionHandler as handler::ConnectionHandler>::OutboundOpenInfo;

    delegate! {
        to self.0 {
            fn listen_protocol(&self) -> SubstreamProtocol<Self::InboundProtocol, Self::InboundOpenInfo>;
            fn poll(&mut self, cx: &mut Context<'_>) -> Poll<ConnectionHandlerEvent<Self::OutboundProtocol, Self::OutboundOpenInfo, Self::ToBehaviour>>;
            fn on_behaviour_event(&mut self, event: Self::FromBehaviour);
            fn on_connection_event(&mut self, event: ConnectionEvent<Self::InboundProtocol, Self::OutboundProtocol, Self::InboundOpenInfo, Self::OutboundOpenInfo>);
        }
    }

    // specifically override this to force connection to stay alive
    fn connection_keep_alive(&self) -> bool {
        true
    }
}
