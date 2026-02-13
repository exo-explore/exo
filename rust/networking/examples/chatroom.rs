use libp2p::identity;
use networking::{self, FromSwarm, ToSwarm};
use tokio::sync::mpsc;
use tokio::{io, io::AsyncBufReadExt as _, select};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init();

    // Configure swarm
    let (to_client, mut from_swarm) = mpsc::channel(20);
    let (to_swarm, from_client) = mpsc::channel(20);
    let mut peer = networking::Peer::new(
        "chatroom!".to_string(),
        identity::Keypair::generate_ed25519(),
        to_client,
        from_client,
    )
    .expect("listen error");

    // Create a Gossipsub topic & subscribe
    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    let jh = tokio::spawn(async move { peer.run().await });
    _ = to_swarm
        .send(ToSwarm::Subscribe("chatting".to_string()))
        .await;

    // Kick it off
    loop {
        select! {
            // on gossipsub outgoing
            Ok(Some(line)) = stdin.next_line() => {
                _ = to_swarm.send(ToSwarm::Message("chatting".to_string(), line.into_bytes())).await;
            }
            event = from_swarm.recv() => match event {
                // on gossipsub incoming
                Some(FromSwarm::Message(peer_id,_, data)) => println!(
                        "\n\nGot message: '{}' from peer: {peer_id}\n\n",
                        String::from_utf8_lossy(&data),
                    ),

                // on discovery
                Some(FromSwarm::Discovered(peer_id)) => {
                    println!("\n\nConnected to: {peer_id}\n\n");
                }
                Some(FromSwarm::Expired(peer_id)) => {
                    println!("\n\nDisconnected from: {peer_id}\n\n");
                }
                Some(FromSwarm::PublishError(e)) => eprintln!("\n\nError {e:?}\n\n"),
                None => break,
            }
        }
    }
    _ = jh.await;
}
