use libp2p::identity;
use networking::swarm::{FromSwarm, Swarm, ToSwarm};
use tokio::sync::mpsc;
use tokio::{io, io::AsyncBufReadExt as _, select};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init();

    let (to_swarm, from_client) = mpsc::channel(20);
    let (to_client, mut from_swarm) = mpsc::channel(20);
    // Configure swarm
    let mut swarm = Swarm::new(
        identity::Keypair::generate_ed25519(),
        from_client,
        to_client,
    )
    .expect("Swarm creation failed");

    // Create a Gossipsub topic & subscribe
    _ = to_swarm
        .send(ToSwarm::Subscribe("test-net".to_owned()))
        .await;

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    tokio::task::spawn(async move { swarm.run().await });

    // Kick it off
    loop {
        select! {
            // on gossipsub outgoing
            Ok(Some(line)) = stdin.next_line() => {
                _= to_swarm.send(ToSwarm::Message("test-net".to_owned(), line.into_bytes())).await;
            }
            event = from_swarm.recv() => match event {
                // on gossipsub incoming
                Some(FromSwarm::Message(pid, topic, content)) => {
                    assert_eq!(topic, "test-net");
                    let fmt = String::from_utf8_lossy(&content);
                    println!("{pid}: {fmt}");
                }

                // on discovery
                Some(FromSwarm::Discovered(pid)) => {
                        eprintln!("\n\nConnected to: {pid}\n\n");
                    }
                Some(FromSwarm::Expired(pid)) => {
                        eprintln!("\n\nDisconnected from: {pid}\n\n");
                }
                None => break,

                // otherwise log any other event
                e => { log::info!("Other event {e:?}"); }
            }
        }
    }
}
