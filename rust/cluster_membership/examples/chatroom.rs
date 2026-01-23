use cluster_membership::Peer;
use libp2p::identity::ed25519::SecretKey;
use tokio::io::{self, AsyncBufReadExt};
use tracing_subscriber::{EnvFilter, filter::LevelFilter};

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init();

    let (mut peer, send, mut recv) =
        Peer::new(SecretKey::generate(), "hello".to_string()).expect("peer should always build");

    let ch = peer.subscribe("chatroom".to_string());
    let jh = tokio::spawn(async move { peer.run().await });

    let mut stdin = io::BufReader::new(io::stdin()).lines();
    loop {
        tokio::select! {
            Ok(Some(line)) = stdin.next_line() => {send.send((ch.clone(), line.into_bytes())).await.expect("example");}
            Some(r) = recv.recv() => match r {
                Ok((_, id, line)) => println!("{:?}:{:?}", id, String::from_utf8_lossy(&line)),
                Err(e) => eprintln!("{e:?}"),
            },
            else => break
        }
    }
    jh.await.expect("task failure");
}
