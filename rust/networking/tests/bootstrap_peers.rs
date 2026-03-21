use futures_lite::StreamExt;
use networking::swarm::{FromSwarm, create_swarm};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Helper: find a free TCP port.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Two nodes connect via bootstrap peers — no mDNS needed.
///
/// Node A listens on a fixed port. Node B bootstraps to A's address.
/// We verify that B emits `FromSwarm::Discovered` for A's peer ID.
#[tokio::test]
async fn two_nodes_connect_via_bootstrap_peers() {
    let port_a = free_port();

    // Node A: listens on a known port, no bootstrap peers
    let keypair_a = libp2p::identity::Keypair::generate_ed25519();
    let peer_id_a = keypair_a.public().to_peer_id();
    let (_tx_a, rx_a) = mpsc::channel(16);
    let swarm_a = create_swarm(keypair_a, rx_a, vec![], port_a).expect("create swarm A");
    let mut stream_a = swarm_a.into_stream();

    // Node B: bootstraps to A's address
    let keypair_b = libp2p::identity::Keypair::generate_ed25519();
    let (_tx_b, rx_b) = mpsc::channel(16);
    let swarm_b = create_swarm(
        keypair_b,
        rx_b,
        vec![format!("/ip4/127.0.0.1/tcp/{port_a}")],
        0,
    )
    .expect("create swarm B");
    let mut stream_b = swarm_b.into_stream();

    // Wait for B to discover A (connection established)
    let connected = timeout(Duration::from_secs(10), async {
        loop {
            tokio::select! {
                Some(event) = stream_a.next() => {
                    // A will also see B connect, but we check from B's perspective
                    let _ = event;
                }
                Some(event) = stream_b.next() => {
                    if let FromSwarm::Discovered { peer_id } = event {
                        if peer_id == peer_id_a {
                            return true;
                        }
                    }
                }
            }
        }
    })
    .await;

    assert!(
        connected.is_ok() && connected.unwrap(),
        "Node B should discover Node A via bootstrap peer"
    );
}

/// Empty bootstrap peers should work (backward compatible).
#[tokio::test]
async fn create_swarm_with_empty_bootstrap_peers() {
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(keypair, rx, vec![], 0);
    assert!(
        swarm.is_ok(),
        "create_swarm with no bootstrap peers should succeed"
    );
}

/// Invalid multiaddr strings are silently filtered out.
#[tokio::test]
async fn create_swarm_ignores_invalid_bootstrap_addrs() {
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(
        keypair,
        rx,
        vec![
            "not-a-valid-multiaddr".to_string(),
            "".to_string(),
            "/ip4/10.0.0.1/tcp/30000".to_string(), // valid
        ],
        0,
    );
    assert!(
        swarm.is_ok(),
        "create_swarm should succeed even with invalid bootstrap addrs"
    );
}

/// Fixed listen port works correctly.
#[tokio::test]
async fn create_swarm_with_fixed_port() {
    let port = free_port();
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let (_tx, rx) = mpsc::channel(16);
    let swarm = create_swarm(keypair, rx, vec![], port);
    assert!(swarm.is_ok(), "create_swarm with fixed port should succeed");
}
