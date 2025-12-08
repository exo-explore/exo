use std::sync::Arc;
use std::time::Duration;

use iroh::SecretKey;
use iroh_gossip::api::{Event, Message};
use iroh_networking::ExoNet;
use n0_future::StreamExt;
use tokio::time::sleep;
use tokio::{io, io::AsyncBufReadExt as _};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init();

    // Configure swarm
    let net = Arc::new(
        ExoNet::init_iroh(SecretKey::generate(&mut rand::rng()), "chatroom")
            .await
            .expect("iroh init shouldn't fail"),
    );
    let innet = Arc::clone(&net);
    let _jh = tokio::spawn(async move { innet.start_auto_dialer().await });

    while net.known_peers.lock().await.is_empty() {
        sleep(Duration::from_secs(1)).await
    }

    // Create a Gossipsub topic & subscribe
    let (send, mut recv) = net
        .subscribe(&"chatting")
        .await
        .expect("topic shouldn't fail");

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    let jh1 = tokio::spawn(async move {
        loop {
            if let Ok(Some(line)) = stdin.next_line().await {
                if let Err(e) = send.broadcast(line.into()).await {
                    println!("Publish error: {e:?}");
                }
            }
        }
    });

    let _ = tokio::spawn(async move {
        while let Some(Ok(event)) = recv.next().await {
            match event {
                // on gossipsub incoming
                Event::Received(Message {
                    content,
                    delivered_from,
                    ..
                }) => println!(
                    "\n\nGot message: '{}' with from peer: {delivered_from}\n\n",
                    String::from_utf8_lossy(&content),
                ),

                // on discovery
                Event::NeighborUp(peer_id) => {
                    println!("\n\nConnected to: {peer_id}\n\n");
                }
                Event::NeighborDown(peer_id) => {
                    eprintln!("\n\nDisconnected from: {peer_id}\n\n");
                }
                Event::Lagged => {
                    eprintln!("\n\nLagged\n\n");
                }
            }
        }
    })
    .await
    .unwrap();
    jh1.await.unwrap();
    _jh.await.unwrap();
}
