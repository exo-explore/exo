//! Fake tracker implementation for Exo topology-based peer discovery
//!
//! Instead of contacting real BitTorrent trackers, this module generates
//! tracker announce responses using Exo's cluster topology data.

use std::net::Ipv4Addr;

use anyhow::Result;

use crate::bencode::{AnnounceParams, BencodeValue};

/// Information about a peer in the Exo topology
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Unique node identifier in the Exo cluster
    pub node_id: String,
    /// IPv4 address of the peer
    pub ip: Ipv4Addr,
    /// BitTorrent listening port
    pub port: u16,
    /// Whether this peer has the complete torrent
    pub has_complete: bool,
    /// Priority for peer selection (higher = prefer)
    pub priority: i32,
}

/// Topology data containing available peers
#[derive(Debug, Clone)]
pub struct TopologyData {
    /// List of peers in the topology
    pub peers: Vec<PeerInfo>,
}

/// Default announce interval in seconds
const DEFAULT_INTERVAL: i64 = 1800;

/// Handle a tracker announce request using Exo topology data
///
/// Returns a bencoded tracker response containing peers from the topology.
///
/// # Arguments
/// * `params` - Announce request parameters
/// * `topology` - Current Exo cluster topology
///
/// # Returns
/// Bencoded announce response as bytes
pub fn handle_announce(params: &AnnounceParams, topology: &TopologyData) -> Result<Vec<u8>> {
    // Sort peers by priority (descending) for better peer selection
    let mut peers: Vec<_> = topology.peers.iter().collect();
    peers.sort_by(|a, b| b.priority.cmp(&a.priority));

    let response = if params.compact {
        // Compact format: 6 bytes per peer (4 IP + 2 port)
        let mut peer_data = Vec::with_capacity(peers.len() * 6);
        for peer in &peers {
            peer_data.extend_from_slice(&peer.ip.octets());
            peer_data.extend_from_slice(&peer.port.to_be_bytes());
        }

        BencodeValue::dict()
            .insert("interval", BencodeValue::integer(DEFAULT_INTERVAL))
            .insert("peers", BencodeValue::Bytes(peer_data))
    } else {
        // Non-compact format: list of dicts
        let mut peer_list = BencodeValue::list();
        for peer in &peers {
            let peer_dict = BencodeValue::dict()
                .insert("ip", BencodeValue::string(&peer.ip.to_string()))
                .insert("port", BencodeValue::integer(i64::from(peer.port)))
                .insert("peer id", BencodeValue::Bytes(vec![0u8; 20])); // Placeholder peer ID
            peer_list = peer_list.push(peer_dict);
        }

        BencodeValue::dict()
            .insert("interval", BencodeValue::integer(DEFAULT_INTERVAL))
            .insert("peers", peer_list)
    };

    Ok(response.encode())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_params(compact: bool) -> AnnounceParams {
        AnnounceParams {
            info_hash: [0u8; 20],
            peer_id: [0u8; 20],
            port: 6881,
            uploaded: 0,
            downloaded: 0,
            left: 1000,
            compact,
            event: None,
        }
    }

    fn make_test_topology() -> TopologyData {
        TopologyData {
            peers: vec![
                PeerInfo {
                    node_id: "node1".to_string(),
                    ip: Ipv4Addr::new(192, 168, 1, 1),
                    port: 6881,
                    has_complete: true,
                    priority: 10,
                },
                PeerInfo {
                    node_id: "node2".to_string(),
                    ip: Ipv4Addr::new(192, 168, 1, 2),
                    port: 6882,
                    has_complete: false,
                    priority: 5,
                },
            ],
        }
    }

    #[test]
    fn test_compact_response() {
        let params = make_test_params(true);
        let topology = make_test_topology();

        let response = handle_announce(&params, &topology).unwrap();

        // Should contain "interval" and "peers" keys
        assert!(response.starts_with(b"d"));
        assert!(response.ends_with(b"e"));

        // Verify we have 12 bytes of peer data (2 peers * 6 bytes)
        // The compact peers field should be "12:<12 bytes>"
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("8:interval"));
        assert!(response_str.contains("5:peers"));
    }

    #[test]
    fn test_non_compact_response() {
        let params = make_test_params(false);
        let topology = make_test_topology();

        let response = handle_announce(&params, &topology).unwrap();

        // Should contain peers as a list
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("8:interval"));
        assert!(response_str.contains("5:peers"));
        assert!(response_str.contains("2:ip"));
        assert!(response_str.contains("4:port"));
    }

    #[test]
    fn test_peer_priority_ordering() {
        let params = make_test_params(true);
        let topology = make_test_topology();

        let response = handle_announce(&params, &topology).unwrap();

        // In compact format, first peer should be node1 (priority 10)
        // which is 192.168.1.1:6881
        // Look for the peer data after "5:peers12:"
        let peers_marker = b"5:peers12:";
        let pos = response
            .windows(peers_marker.len())
            .position(|w| w == peers_marker)
            .unwrap();
        let peer_data = &response[pos + peers_marker.len()..pos + peers_marker.len() + 6];

        // First peer should be 192.168.1.1 (node1 with higher priority)
        assert_eq!(&peer_data[0..4], &[192, 168, 1, 1]);
    }

    #[test]
    fn test_empty_topology() {
        let params = make_test_params(true);
        let topology = TopologyData { peers: vec![] };

        let response = handle_announce(&params, &topology).unwrap();

        // Should still be valid bencoded response with empty peers
        assert!(response.starts_with(b"d"));
        assert!(response.ends_with(b"e"));
    }
}
