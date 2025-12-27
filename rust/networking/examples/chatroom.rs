#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cargo)]

use std::sync::Arc;
use std::time::Duration;

use iroh::SecretKey;
use iroh_gossip::api::{Event, Message};
use n0_future::StreamExt as _;
use networking::ExoNet;
use tokio::time::sleep;
use tokio::{io, io::AsyncBufReadExt as _};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init()
        .expect("logger");

    // Configure swarm
    let net = Arc::new(
        ExoNet::init_iroh(SecretKey::generate(&mut rand::rng()), "chatroom")
            .await
            .expect("iroh init shouldn't fail"),
    );
    let innet = Arc::clone(&net);
    let jh1 = tokio::spawn(async move { innet.start_auto_dialer().await });

    while net.known_peers.lock().await.is_empty() {
        sleep(Duration::from_secs(1)).await;
    }

    // Create a Gossipsub topic & subscribe
    let (send, mut recv) = net
        .subscribe("chatting")
        .await
        .expect("topic shouldn't fail");

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    let jh2 = tokio::spawn(async move {
        loop {
            if let Ok(Some(line)) = stdin.next_line().await
                && let Err(e) = send.broadcast(line.into()).await
            {
                println!("Publish error: {e:?}");
            }
        }
    });

    tokio::spawn(async move {
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
    jh2.await.unwrap();
}
