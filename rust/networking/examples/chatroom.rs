use futures_lite::StreamExt as _;
use libp2p::identity;
use networking::{FromSwarm, ToSwarm};
use tokio::sync::{mpsc, oneshot};
use tokio::{io, io::AsyncBufReadExt as _};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    let (to_swarm, from_client) = mpsc::channel(20);

    // Configure swarm
    let mut swarm = networking::create_swarm(identity::Keypair::generate_ed25519(), from_client)
        .expect("Swarm creation failed")
        .into_stream();

    // Create a Gossipsub topic & subscribe
    let (sub_tx, sub_rx) = oneshot::channel();
    to_swarm
        .send(ToSwarm::Subscribe {
            topic: "test-net".to_owned(),
            result_sender: sub_tx,
        })
        .await
        .expect("should send");

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    tokio::task::spawn(async move {
        sub_rx
            .await
            .expect("tx not dropped")
            .expect("subscribe shouldn't fail");
        loop {
            if let Ok(Some(line)) = stdin.next_line().await {
                let (tx, rx) = oneshot::channel();
                if let Err(e) = to_swarm
                    .send(ToSwarm::Publish {
                        topic: "test-net".to_owned(),
                        data: line.as_bytes().to_vec(),
                        result_sender: tx,
                    })
                    .await
                {
                    println!("Send error: {e:?}");
                    return;
                }
                match rx.await {
                    Ok(Err(e)) => println!("Publish error: {e:?}"),
                    Err(e) => println!("Publish error: {e:?}"),
                    Ok(_) => {}
                }
            }
        }
    });

    // Kick it off
    loop {
        // on gossipsub outgoing
        match swarm.next().await {
            // on gossipsub incoming
            Some(FromSwarm::Discovered { peer_id }) => {
                println!("\n\nconnected to {peer_id}\n\n")
            }
            Some(FromSwarm::Expired { peer_id }) => {
                println!("\n\ndisconnected from {peer_id}\n\n")
            }
            Some(FromSwarm::Message { from, topic, data }) => {
                println!("{topic}/{from}:\n{}", String::from_utf8_lossy(&data))
            }
            None => {}
        }
    }
}
