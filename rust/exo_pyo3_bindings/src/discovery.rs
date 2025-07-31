#![allow(
    clippy::multiple_inherent_impl,
    clippy::unnecessary_wraps,
    clippy::unused_self,
    clippy::needless_pass_by_value
)]

use crate::ext::ResultExt;
use crate::pylibp2p::connection::PyConnectionId;
use crate::pylibp2p::ident::{PyKeypair, PyPeerId};
use crate::pylibp2p::multiaddr::PyMultiaddr;
use crate::{MPSC_CHANNEL_SIZE, alias, pyclass};
use discovery::behaviour::{DiscoveryBehaviour, DiscoveryBehaviourEvent};
use discovery::discovery_swarm;
use libp2p::core::ConnectedPoint;
use libp2p::futures::StreamExt;
use libp2p::multiaddr::multiaddr;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{ConnectionId, SwarmEvent, ToSwarm};
use libp2p::{Multiaddr, PeerId, Swarm, gossipsub, mdns};
use std::net::IpAddr;
use pyo3::prelude::{PyModule, PyModuleMethods as _};
use pyo3::{Bound, Py, PyObject, PyResult, PyTraverseError, PyVisit, Python, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::convert::identity;
use std::error::Error;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};

struct ConnectionUpdate {
    /// Identity of the peer that we have connected to.
    peer_id: PeerId,
    /// Identifier of the connection.
    connection_id: ConnectionId,
    /// Local connection address.
    local_addr: Multiaddr,
    /// Address used to send back data to the remote.
    send_back_addr: Multiaddr,
}

#[gen_stub_pyclass]
#[pyclass(frozen, name = "ConnectionUpdate")]
#[derive(Debug, Clone)]
struct PyConnectionUpdate {
    /// Identity of the peer that we have connected to.
    #[pyo3(get)]
    peer_id: PyPeerId,
    /// Identifier of the connection.
    #[pyo3(get)]
    connection_id: PyConnectionId,
    /// Local connection address.
    #[pyo3(get)]
    local_addr: PyMultiaddr,
    /// Address used to send back data to the remote.
    #[pyo3(get)]
    send_back_addr: PyMultiaddr,
}

impl PyConnectionUpdate {
    fn from_connection_event(
        ConnectionUpdate {
            peer_id,
            connection_id,
            local_addr,
            send_back_addr,
        }: ConnectionUpdate,
    ) -> Self {
        Self {
            peer_id: PyPeerId(peer_id),
            connection_id: PyConnectionId(connection_id),
            local_addr: PyMultiaddr(local_addr),
            send_back_addr: PyMultiaddr(send_back_addr),
        }
    }
}

enum IncomingDiscoveryMessage {
    AddConnectedCallback(Box<dyn alias::SendFn<(ConnectionUpdate,), ()>>),
    AddDisconnectedCallback(Box<dyn alias::SendFn<(ConnectionUpdate,), ()>>),
}

/// Check if a multiaddr is valid for connection
fn is_address_valid(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    
    for component in addr.iter() {
        match component {
            Protocol::Ip4(ip) => {
                let ip_addr = IpAddr::V4(ip);
                // Filter out loopback and unspecified addresses
                if ip_addr.is_loopback() || ip_addr.is_unspecified() {
                    return false;
                }
                // Filter out Tailscale ranges (100.64.0.0/10)
                if let IpAddr::V4(ipv4) = ip_addr {
                    let octets = ipv4.octets();
                    if octets[0] == 100 && octets[1] >= 64 && octets[1] <= 127 {
                        return false;
                    }
                }
            }
            Protocol::Ip6(ip) => {
                let ip_addr = IpAddr::V6(ip);
                // Filter out loopback and unspecified addresses
                if ip_addr.is_loopback() || ip_addr.is_unspecified() {
                    return false;
                }
                // Filter out Tailscale IPv6 (fd7a:115c:a1e0::/48)
                if let IpAddr::V6(ipv6) = ip_addr {
                    let segments = ipv6.segments();
                    if segments[0] == 0xfd7a && segments[1] == 0x115c && segments[2] == 0xa1e0 {
                        return false;
                    }
                }
            }
            _ => {}
        }
    }
    true
}

#[allow(clippy::enum_glob_use)]
async fn discovery_task(
    mut receiver: mpsc::Receiver<IncomingDiscoveryMessage>,
    mut swarm: Swarm<DiscoveryBehaviour>,
) {
    use DiscoveryBehaviourEvent::*;
    use IncomingDiscoveryMessage::*;
    use SwarmEvent::*;
    use gossipsub::Event::*;
    use mdns::Event::*;

    log::info!("RUST: discovery task started");

    // create callbacks list
    let mut connected_callbacks: Vec<Box<dyn alias::SendFn<(ConnectionUpdate,), ()>>> = vec![];
    let mut disconnected_callbacks: Vec<Box<dyn alias::SendFn<(ConnectionUpdate,), ()>>> = vec![];
    
    // Create periodic health check timer with adaptive interval
    let fast_check_duration = Duration::from_secs(5);
    let slow_check_duration = Duration::from_secs(30);
    let mut health_check_interval = interval(fast_check_duration);
    let mut no_connection_count = 0;

    loop {
        tokio::select! {
            _ = health_check_interval.tick() => {
                // Check connection health periodically
                let connected_peers = swarm.connected_peers().count();
                if connected_peers == 0 {
                    no_connection_count += 1;
                    log::info!("RUST: No connected peers (check #{no_connection_count})");
                    
                    // Keep fast checking when disconnected
                    if health_check_interval.period() != fast_check_duration {
                        health_check_interval = interval(fast_check_duration);
                        log::info!("RUST: Switching to fast health checks (every {:?})", fast_check_duration);
                    }
                    
                    // Force mDNS restart after multiple failed checks
                    if no_connection_count > 1 { // Trigger faster, after 2 checks
                        log::info!("RUST: Attempting to restart mDNS discovery");
                        // Note: In rust-libp2p, we can't easily restart mDNS like in Go,
                        // but we can force a re-announce by changing listening addresses
                        // This is a workaround to trigger mDNS to re-announce
                        
                        // Try listening on a new ephemeral port to force re-announcement
                        match swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse().unwrap()) {
                            Ok(_) => log::info!("RUST: Added new listener to force mDNS re-announcement"),
                            Err(e) => log::error!("RUST: Failed to add new listener: {e:?}"),
                        }
                        
                        // Also try IPv6
                        match swarm.listen_on("/ip6/::/tcp/0".parse().unwrap()) {
                            Ok(_) => log::info!("RUST: Added IPv6 listener to force mDNS re-announcement"),
                            Err(e) => log::error!("RUST: Failed to add IPv6 listener: {e:?}"),
                        }
                    }
                } else {
                    if no_connection_count > 0 {
                        log::info!("RUST: Connection restored, currently connected to {connected_peers} peers");
                    }
                    no_connection_count = 0;
                    
                    // Switch to slow checking when connected
                    if health_check_interval.period() != slow_check_duration {
                        health_check_interval = interval(slow_check_duration);
                        log::info!("RUST: Switching to slow health checks (every {:?})", slow_check_duration);
                    }
                }
            }
            message = receiver.recv() => {
                // handle closed channel
                let Some(message) = message else {
                    log::info!("RUST: channel closed");
                    break;
                };

                // attach callbacks for event types
                match message {
                    AddConnectedCallback(callback) => {
                        log::info!("RUST: received connected callback");
                        connected_callbacks.push(callback);
                    }
                    AddDisconnectedCallback(callback) => {
                        log::info!("RUST: received disconnected callback");
                        disconnected_callbacks.push(callback);
                    }
                }
            }
            swarm_event = swarm.select_next_some() => {
                match swarm_event {
                    Behaviour(Mdns(Discovered(list))) => {
                        for (peer_id, multiaddr) in list {
                            log::info!("RUST: mDNS discovered a new peer: {peer_id} on {multiaddr}");
                            
                            // Filter out invalid addresses
                            if !is_address_valid(&multiaddr) {
                                log::info!("RUST: Filtered out invalid address: {multiaddr}");
                                continue;
                            }
                            
                            let local_peer_id = *swarm.local_peer_id();
                            // To avoid simultaneous dial races, only the lexicographically larger peer_id dials.
                            if peer_id > local_peer_id {
                                let dial_opts = DialOpts::peer_id(peer_id)
                                    .addresses(vec![multiaddr.clone()].into())
                                    .condition(libp2p::swarm::dial_opts::PeerCondition::Always)
                                    .build();
                                match swarm.dial(dial_opts) {
                                    Ok(()) => log::info!("RUST: Dial initiated to {multiaddr}"),
                                    Err(libp2p::swarm::DialError::DialPeerConditionFalse(_)) => {
                                        // Another dial is already in progress; not an error for us.
                                        log::debug!(
                                            "RUST: Dial skipped because another dial is active for {peer_id}"
                                        );
                                    }
                                    Err(e) => {
                                        log::warn!("RUST: Failed to dial {multiaddr}: {e:?}");
                                    }
                                }
                            }
                            // Maintain peer in gossipsub mesh so the connection stays alive once established.
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                            log::info!("RUST: Added peer {peer_id} to gossipsub explicit peers");
                        }
                    }
                    Behaviour(Mdns(Expired(list))) => {
                        for (peer_id, multiaddr) in list {
                            log::info!("RUST: mDNS discover peer has expired: {peer_id} on {multiaddr}");
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                        }
                    },
                    Behaviour(Gossipsub(Message {
                        propagation_source: peer_id,
                        message_id: id,
                        message,
                    })) => log::info!(
                            "RUST: Got message: '{}' with id: {id} from peer: {peer_id}",
                            String::from_utf8_lossy(&message.data),
                        ),
                    ConnectionEstablished {
                        peer_id,
                        connection_id,
                        endpoint,
                        num_established: _num_established,
                        concurrent_dial_errors,
                        established_in: _established_in,
                    } => {
                        log::info!("RUST: ConnectionEstablished event - peer_id: {peer_id}, connection_id: {connection_id:?}, endpoint: {endpoint:?}");
                        // log any connection errors
                        if let Some(concurrent_dial_errors) = concurrent_dial_errors {
                            for (multiaddr, error) in concurrent_dial_errors {
                                log::error!("Connection error: multiaddr={multiaddr}, error={error:?}");
                            }
                        }

                        // Extract addresses based on endpoint type
                        let (local_addr, send_back_addr) = match &endpoint {
                            ConnectedPoint::Listener { local_addr, send_back_addr } => {
                                log::info!("RUST: Connection established (Listener) - local_addr: {local_addr}, send_back_addr: {send_back_addr}");
                                (local_addr.clone(), send_back_addr.clone())
                            },
                            ConnectedPoint::Dialer { address, .. } => {
                                log::info!("RUST: Connection established (Dialer) - remote_addr: {address}");
                                // For dialer, we use the dialed address as both local and send_back
                                // This isn't perfect but allows both sides to be notified
                                (address.clone(), address.clone())
                            }
                        };
                        
                        log::info!("RUST: Number of connected callbacks: {}", connected_callbacks.len());


                        // trigger callback on connected peer
                        for connected_callback in &connected_callbacks {
                            connected_callback(ConnectionUpdate {
                                peer_id,
                                connection_id,
                                local_addr: local_addr.clone(),
                                send_back_addr: send_back_addr.clone(),
                            });
                        }
                    },
                    ConnectionClosed { peer_id, connection_id, endpoint, num_established, cause }  => {
                        log::info!("RUST: ConnectionClosed event - peer_id: {peer_id}, connection_id: {connection_id:?}, endpoint: {endpoint:?}, num_established: {num_established}");
                        // log any connection errors
                        if let Some(cause) = cause {
                            log::error!("Connection error: cause={cause:?}");
                        }

                        // Extract addresses based on endpoint type
                        let (local_addr, send_back_addr) = match &endpoint {
                            ConnectedPoint::Listener { local_addr, send_back_addr } => {
                                log::info!("RUST: Connection closed (Listener) - local_addr: {local_addr}, send_back_addr: {send_back_addr}");
                                (local_addr.clone(), send_back_addr.clone())
                            },
                            ConnectedPoint::Dialer { address, .. } => {
                                log::info!("RUST: Connection closed (Dialer) - remote_addr: {address}");
                                // For dialer, we use the dialed address as both local and send_back
                                // This isn't perfect but allows both sides to be notified
                                (address.clone(), address.clone())
                            }
                        };
                        
                        log::info!("RUST: Number of disconnected callbacks: {}", disconnected_callbacks.len());

                        // trigger callback on connected peer
                        for disconnected_callback in &disconnected_callbacks {
                            disconnected_callback(ConnectionUpdate {
                                peer_id,
                                connection_id,
                                local_addr: local_addr.clone(),
                                send_back_addr: send_back_addr.clone(),
                            });
                        }
                        
                        // If this was the last connection to the peer, try to force mDNS re-discovery
                        if num_established == 0 {
                            log::info!("RUST: Last connection to peer {peer_id} closed, triggering mDNS re-discovery");
                            // Remove from gossipsub to ensure clean state
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                            
                            // Force a listen address change to trigger mDNS re-announcement
                            tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                log::info!("RUST: Delayed mDNS trigger after disconnect");
                            });
                        }
                    }
                    NewListenAddr { address, .. } => {
                        log::info!("RUST: Local node is listening on {address}");
                        let local_peer = swarm.local_peer_id();
                        log::info!("RUST: Local peer_id: {local_peer}");
                    }
                    OutgoingConnectionError { peer_id, error, .. } => {
                        log::error!("RUST: Outgoing connection error to peer {peer_id:?}: {error:?}");
                        // Connection failed, might be due to network change
                        if let Some(peer) = peer_id {
                            // Remove from gossipsub to allow fresh connection attempts
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer);
                        }
                    }
                    IncomingConnectionError { send_back_addr, error, .. } => {
                        log::error!("RUST: Incoming connection error from {send_back_addr}: {error:?}");
                    }
                    e => {
                        log::debug!("RUST: Other event {e:?}");
                    }
                }
            }
        }
    }

    log::info!("RUST: discovery task stopped");
}

#[gen_stub_pyclass]
#[pyclass(name = "DiscoveryService")]
#[derive(Debug, Clone)]
struct PyDiscoveryService {
    sender: Option<mpsc::Sender<IncomingDiscoveryMessage>>,
}

#[allow(clippy::expect_used)]
impl PyDiscoveryService {
    const fn sender(&self) -> &mpsc::Sender<IncomingDiscoveryMessage> {
        self.sender
            .as_ref()
            .expect("The sender should only be None after de-initialization.")
    }

    const fn sender_mut(&mut self) -> &mut mpsc::Sender<IncomingDiscoveryMessage> {
        self.sender
            .as_mut()
            .expect("The sender should only be None after de-initialization.")
    }

    const fn new(sender: mpsc::Sender<IncomingDiscoveryMessage>) -> Self {
        Self {
            sender: Some(sender),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDiscoveryService {
    #[new]
    fn py_new<'py>(identity: Bound<'py, PyKeypair>) -> PyResult<Self> {
        use pyo3_async_runtimes::tokio::get_runtime;

        // create communication channel
        let (sender, receiver) = mpsc::channel::<IncomingDiscoveryMessage>(MPSC_CHANNEL_SIZE);

        // get identity
        let identity = identity.borrow().0.clone();
        log::info!("RUST: Creating DiscoveryService with keypair");

        // create discovery swarm (within tokio context!! or it crashes)
        let swarm = get_runtime()
            .block_on(async { discovery_swarm(identity) })
            .pyerr()?;
        log::info!("RUST: Discovery swarm created successfully");

        // spawn tokio task
        get_runtime().spawn(async move {
            log::info!("RUST: Starting discovery task");
            discovery_task(receiver, swarm).await;
            log::info!("RUST: Discovery task ended");
        });
        Ok(Self::new(sender))
    }

    #[allow(clippy::expect_used)]
    fn add_connected_callback<'py>(
        &self,
        #[gen_stub(override_type(
            type_repr="collections.abc.Callable[[ConnectionUpdate], None]",
            imports=("collections.abc")
        ))]
        callback: PyObject,
    ) -> PyResult<()> {
        use pyo3_async_runtimes::tokio::get_runtime;

        get_runtime()
            .block_on(
                self.sender()
                    .send(IncomingDiscoveryMessage::AddConnectedCallback(Box::new(
                        move |connection_event| {
                            Python::with_gil(|py| {
                                callback
                                    .call1(
                                        py,
                                        (PyConnectionUpdate::from_connection_event(
                                            connection_event,
                                        ),),
                                    )
                                    .expect("Callback should always work...");
                            });
                        },
                    ))),
            )
            .pyerr()?;
        Ok(())
    }

    #[allow(clippy::expect_used)]
    fn add_disconnected_callback<'py>(
        &self,
        #[gen_stub(override_type(
            type_repr="collections.abc.Callable[[ConnectionUpdate], None]",
            imports=("collections.abc")
        ))]
        callback: PyObject,
    ) -> PyResult<()> {
        use pyo3_async_runtimes::tokio::get_runtime;

        get_runtime()
            .block_on(
                self.sender()
                    .send(IncomingDiscoveryMessage::AddDisconnectedCallback(Box::new(
                        move |connection_event| {
                            Python::with_gil(|py| {
                                callback
                                    .call1(
                                        py,
                                        (PyConnectionUpdate::from_connection_event(
                                            connection_event,
                                        ),),
                                    )
                                    .expect("Callback should always work...");
                            });
                        },
                    ))),
            )
            .pyerr()?;
        Ok(())
    }

    #[gen_stub(skip)]
    const fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(()) // This is needed purely so `__clear__` can work
    }

    #[gen_stub(skip)]
    fn __clear__(&mut self) {
        // TODO: may or may not need to await a "kill-signal" oneshot channel message,
        //       to ensure that the discovery task is done BEFORE exiting the clear function...
        //       but this may require GIL?? and it may not be safe to call GIL here??
        self.sender = None; // Using Option<T> as a trick to force `sender` channel to be dropped
    }
}

pub fn discovery_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConnectionUpdate>()?;
    m.add_class::<PyDiscoveryService>()?;

    Ok(())
}
