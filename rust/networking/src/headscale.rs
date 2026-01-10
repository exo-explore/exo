//! Headscale-based peer discovery for exo.
//!
//! This module provides peer discovery via the Headscale API, allowing exo nodes
//! running on different networks to discover each other through a Headscale coordination server.

use crate::ext::MultiaddrExt;
use futures::FutureExt;
use futures_timer::Delay;
use libp2p::core::transport::PortUse;
use libp2p::core::{ConnectedPoint, Endpoint};
use libp2p::swarm::behaviour::ConnectionEstablished;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{
    dummy, ConnectionClosed, ConnectionDenied, ConnectionId, FromSwarm, NetworkBehaviour, THandler,
    THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use libp2p::{Multiaddr, PeerId};
use reqwest::Client;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::io;
use std::net::IpAddr;
use std::str::FromStr;
use std::task::{Context, Poll};
use std::time::Duration;
use util::wakerdeque::WakerDeque;

/// Default interval for polling the Headscale API
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Default port that exo listens on (this should be configurable)
const DEFAULT_EXO_PORT: u16 = 52415;

/// Retry interval for connecting to discovered peers
const RETRY_CONNECT_INTERVAL: Duration = Duration::from_secs(5);

/// Configuration for Headscale discovery
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL for the Headscale API (e.g., "https://headscale.example.com")
    pub api_base_url: String,
    /// API key for authenticating with the Headscale API
    pub api_key: String,
    /// How often to poll the Headscale API for peer updates
    pub poll_interval: Duration,
    /// The port that exo is listening on (for constructing peer addresses)
    pub exo_port: u16,
    /// Optional list of node names to filter (if empty, all nodes are discovered)
    pub allowed_nodes: Vec<String>,
}

impl Config {
    pub fn new(api_base_url: String, api_key: String) -> Self {
        Self {
            api_base_url,
            api_key,
            poll_interval: DEFAULT_POLL_INTERVAL,
            exo_port: DEFAULT_EXO_PORT,
            allowed_nodes: Vec::new(),
        }
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn with_exo_port(mut self, port: u16) -> Self {
        self.exo_port = port;
        self
    }

    pub fn with_allowed_nodes(mut self, nodes: Vec<String>) -> Self {
        self.allowed_nodes = nodes;
        self
    }
}

/// A node returned from the Headscale API
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadscaleNode {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub ip_addresses: Vec<String>,
    #[serde(default)]
    pub online: bool,
    #[serde(default)]
    pub forced_tags: Vec<String>,
}

/// Response from the Headscale API when listing nodes
#[derive(Debug, Clone, Deserialize)]
pub struct ListNodesResponse {
    pub nodes: Vec<HeadscaleNode>,
}

/// Exo-specific attributes stored in Headscale node tags
#[derive(Debug, Clone, Default)]
pub struct ExoNodeAttributes {
    /// The libp2p peer ID (base58 encoded)
    pub peer_id: Option<String>,
    /// The port exo is listening on
    pub port: Option<u16>,
}

impl ExoNodeAttributes {
    /// Parse exo attributes from Headscale forced tags
    /// Tags are expected to be in format: "tag:exo_key=value"
    pub fn from_tags(tags: &[String]) -> Self {
        let mut attrs = Self::default();
        for tag in tags {
            if let Some(kv) = tag.strip_prefix("tag:exo_") {
                if let Some((key, value)) = kv.split_once('=') {
                    match key {
                        "peer_id" => attrs.peer_id = Some(value.to_string()),
                        "port" => attrs.port = value.parse().ok(),
                        _ => {}
                    }
                }
            }
        }
        attrs
    }
}

/// Events emitted by the Headscale discovery behaviour
#[derive(Debug, Clone)]
pub enum Event {
    /// A new peer was discovered via Headscale
    Discovered {
        peer_id: PeerId,
        addresses: Vec<Multiaddr>,
    },
    /// A previously discovered peer is no longer available
    Expired { peer_id: PeerId },
    /// A connection to a Headscale-discovered peer was established
    ConnectionEstablished {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
    /// A connection to a Headscale-discovered peer was closed
    ConnectionClosed {
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    },
}

/// State for a discovered peer
#[derive(Debug, Clone)]
struct DiscoveredPeer {
    peer_id: PeerId,
    addresses: Vec<Multiaddr>,
    connected: bool,
}

/// Headscale discovery behaviour for libp2p
pub struct Behaviour {
    config: Config,
    http_client: Client,

    /// Our own peer ID (to filter ourselves from discovered nodes)
    local_peer_id: PeerId,

    /// Currently discovered peers from Headscale
    discovered_peers: HashMap<PeerId, DiscoveredPeer>,

    /// Timer for polling the Headscale API
    poll_timer: Delay,

    /// Timer for retrying connections to discovered peers
    retry_timer: Delay,

    /// Pending events to emit
    pending_events: WakerDeque<ToSwarm<Event, Infallible>>,

    /// Flag to indicate we need to poll the API
    needs_poll: bool,
}

impl Behaviour {
    /// Create a new Headscale discovery behaviour
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created or if the API URL is invalid.
    pub fn new(config: Config, local_peer_id: PeerId) -> io::Result<Self> {
        // Validate the API URL
        if config.api_base_url.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Headscale API URL cannot be empty",
            ));
        }
        if !config.api_base_url.starts_with("http://")
            && !config.api_base_url.starts_with("https://")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Headscale API URL must start with http:// or https://",
            ));
        }
        if !config.api_base_url.starts_with("https://") {
            log::warn!(
                "Headscale API URL uses HTTP instead of HTTPS. \
                 Consider using HTTPS for secure API key transmission."
            );
        }

        let http_client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(Self {
            poll_timer: Delay::new(Duration::ZERO), // Poll immediately on start
            retry_timer: Delay::new(RETRY_CONNECT_INTERVAL),
            config,
            http_client,
            local_peer_id,
            discovered_peers: HashMap::new(),
            pending_events: WakerDeque::new(),
            needs_poll: true,
        })
    }

    /// Fetch nodes from the Headscale API
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails (network error, timeout, etc.)
    /// - The server returns a non-2xx status code (401 unauthorized, 403 forbidden, 5xx server error)
    /// - The response body cannot be parsed as JSON
    async fn fetch_nodes(
        client: &Client,
        api_base_url: &str,
        api_key: &str,
    ) -> Result<Vec<HeadscaleNode>, reqwest::Error> {
        let url = format!("{}/api/v1/node", api_base_url);
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?
            .error_for_status()?;

        let list_response: ListNodesResponse = response.json().await?;
        Ok(list_response.nodes)
    }

    /// Process nodes fetched from Headscale and update our discovered peers
    fn process_nodes(&mut self, nodes: Vec<HeadscaleNode>) {
        let mut seen_peers = HashSet::new();

        for node in nodes {
            // Skip offline nodes
            if !node.online {
                continue;
            }

            // Skip if we're filtering by allowed nodes and this node isn't in the list
            if !self.config.allowed_nodes.is_empty()
                && !self.config.allowed_nodes.contains(&node.name)
            {
                continue;
            }

            // Parse exo attributes from tags
            let attrs = ExoNodeAttributes::from_tags(&node.forced_tags);

            // Get the peer ID from tags, or skip this node
            let peer_id = match attrs
                .peer_id
                .as_ref()
                .and_then(|s| PeerId::from_str(s).ok())
            {
                Some(id) => id,
                None => {
                    log::debug!(
                        "Headscale node {} doesn't have a valid exo peer_id tag, skipping",
                        node.name
                    );
                    continue;
                }
            };

            // Skip ourselves
            if peer_id == self.local_peer_id {
                continue;
            }

            // Get the port from tags or use default
            let port = attrs.port.unwrap_or(self.config.exo_port);

            // Build multiaddresses from IP addresses
            let addresses: Vec<Multiaddr> = node
                .ip_addresses
                .iter()
                .filter_map(|ip_str| {
                    if let Ok(ip) = ip_str.parse::<IpAddr>() {
                        let addr_str = match ip {
                            IpAddr::V4(v4) => format!("/ip4/{}/tcp/{}", v4, port),
                            IpAddr::V6(v6) => format!("/ip6/{}/tcp/{}", v6, port),
                        };
                        addr_str.parse().ok()
                    } else {
                        None
                    }
                })
                .collect();

            if addresses.is_empty() {
                log::debug!(
                    "Headscale node {} has no valid IP addresses, skipping",
                    node.name
                );
                continue;
            }

            seen_peers.insert(peer_id);

            // Check if this is a new peer or if addresses changed
            let is_new = if let Some(existing) = self.discovered_peers.get(&peer_id) {
                existing.addresses != addresses
            } else {
                true
            };

            if is_new {
                log::info!(
                    "Discovered exo peer {} ({}) via Headscale at {:?}",
                    peer_id,
                    node.name,
                    addresses
                );

                // Emit discovered event
                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::Discovered {
                        peer_id,
                        addresses: addresses.clone(),
                    }));

                // Dial the peer with all addresses in a single request
                self.pending_events.push_back(ToSwarm::Dial {
                    opts: DialOpts::peer_id(peer_id)
                        .addresses(addresses.clone())
                        .build(),
                });

                // Update or insert the peer
                self.discovered_peers.insert(
                    peer_id,
                    DiscoveredPeer {
                        peer_id,
                        addresses,
                        connected: false,
                    },
                );
            }
        }

        // Find peers that are no longer in the Headscale list
        let expired_peers: Vec<PeerId> = self
            .discovered_peers
            .keys()
            .filter(|id| !seen_peers.contains(id))
            .cloned()
            .collect();

        for peer_id in expired_peers {
            log::info!("Exo peer {} is no longer available via Headscale", peer_id);
            self.discovered_peers.remove(&peer_id);
            self.pending_events
                .push_back(ToSwarm::GenerateEvent(Event::Expired { peer_id }));
        }
    }

    fn on_connection_established(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // Update connection state if this is a Headscale-discovered peer
        if let Some(peer) = self.discovered_peers.get_mut(&peer_id) {
            peer.connected = true;
        }

        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn on_connection_closed(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        // Update connection state if this is a Headscale-discovered peer
        if let Some(peer) = self.discovered_peers.get_mut(&peer_id) {
            peer.connected = false;
        }

        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = dummy::ConnectionHandler;
    type ToSwarm = Event;

    fn handle_pending_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _local_addr: &Multiaddr,
        _remote_addr: &Multiaddr,
    ) -> Result<(), ConnectionDenied> {
        Ok(())
    }

    fn handle_pending_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _maybe_peer: Option<PeerId>,
        _addresses: &[Multiaddr],
        _effective_role: Endpoint,
    ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
        Ok(vec![])
    }

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _local_addr: &Multiaddr,
        _remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(dummy::ConnectionHandler)
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(dummy::ConnectionHandler)
    }

    fn on_connection_handler_event(
        &mut self,
        _peer_id: PeerId,
        _connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        libp2p::core::util::unreachable(event)
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                // Only track connections to Headscale-discovered peers
                if self.discovered_peers.contains_key(&peer_id) {
                    let remote_address = match endpoint {
                        ConnectedPoint::Dialer { address, .. } => address,
                        ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                    };

                    if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                        self.on_connection_established(peer_id, connection_id, ip, port);
                    }
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                // Only track disconnections from Headscale-discovered peers
                if self.discovered_peers.contains_key(&peer_id) {
                    let remote_address = match endpoint {
                        ConnectedPoint::Dialer { address, .. } => address,
                        ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                    };

                    if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                        self.on_connection_closed(peer_id, connection_id, ip, port);
                    }
                }
            }
            // Other FromSwarm events (e.g., DialFailure, ListenFailure) are not relevant
            // for Headscale discovery as we only track successful connections.
            _ => {}
        }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        // Check if we need to poll the Headscale API
        if self.poll_timer.poll_unpin(cx).is_ready() || self.needs_poll {
            self.needs_poll = false;

            // TODO: Refactor to use proper async task spawning with a channel-based approach.
            // Currently using block_in_place which is not ideal as it can cause executor
            // starvation under heavy load. A better approach would be to:
            // 1. Spawn a background task that periodically fetches nodes
            // 2. Send results through an mpsc channel
            // 3. Have poll() check the channel for new results
            // This would align with libp2p's non-blocking poll() contract.
            let client = self.http_client.clone();
            let api_base_url = self.config.api_base_url.clone();
            let api_key = self.config.api_key.clone();
            match tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(Self::fetch_nodes(
                    &client,
                    &api_base_url,
                    &api_key,
                ))
            }) {
                Ok(nodes) => {
                    log::debug!("Fetched {} nodes from Headscale", nodes.len());
                    self.process_nodes(nodes);
                }
                Err(e) => {
                    log::warn!("Failed to fetch nodes from Headscale: {}", e);
                }
            }

            // Reset the poll timer
            self.poll_timer.reset(self.config.poll_interval);
        }

        // Retry connecting to discovered but not connected peers
        if self.retry_timer.poll_unpin(cx).is_ready() {
            for peer in self.discovered_peers.values() {
                if !peer.connected {
                    self.pending_events.push_back(ToSwarm::Dial {
                        opts: DialOpts::peer_id(peer.peer_id)
                            .addresses(peer.addresses.clone())
                            .build(),
                    });
                }
            }
            self.retry_timer.reset(RETRY_CONNECT_INTERVAL);
        }

        // Return any pending events
        if let Some(event) = self.pending_events.pop_front(cx) {
            return Poll::Ready(event);
        }

        Poll::Pending
    }
}
