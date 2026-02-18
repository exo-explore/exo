use futures::stream::StreamExt as _;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use networking::{discovery, swarm};
use tokio::{io, io::AsyncBufReadExt as _, select};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .try_init();

    // Configure swarm
    let mut swarm =
        swarm::create_swarm(identity::Keypair::generate_ed25519()).expect("Swarm creation failed");

    // Create a Gossipsub topic & subscribe
    let topic = gossipsub::IdentTopic::new("test-net");
    swarm
        .behaviour_mut()
        .gossipsub
        .subscribe(&topic)
        .expect("Subscribing to topic failed");

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    // Kick it off
    loop {
        select! {
            // on gossipsub outgoing
            Ok(Some(line)) = stdin.next_line() => {
                if let Err(e) = swarm
                    .behaviour_mut().gossipsub
                    .publish(topic.clone(), line.as_bytes()) {
                    println!("Publish error: {e:?}");
                }
            }
            event = swarm.select_next_some() => match event {
                // on gossipsub incoming
                SwarmEvent::Behaviour(swarm::BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source: peer_id,
                    message_id: id,
                    message,
                })) => println!(
                        "\n\nGot message: '{}' with id: {id} from peer: {peer_id}\n\n",
                        String::from_utf8_lossy(&message.data),
                    ),

                // on discovery
                SwarmEvent::Behaviour(swarm::BehaviourEvent::Discovery(e)) => match e {
                    discovery::Event::ConnectionEstablished {
                        peer_id, connection_id, remote_ip, remote_tcp_port
                    } => {
                        println!("\n\nConnected to: {peer_id}; connection ID: {connection_id}; remote IP: {remote_ip}; remote TCP port: {remote_tcp_port}\n\n");
                    }
                    discovery::Event::ConnectionClosed {
                        peer_id, connection_id, remote_ip, remote_tcp_port
                    } => {
                        eprintln!("\n\nDisconnected from: {peer_id}; connection ID: {connection_id}; remote IP: {remote_ip}; remote TCP port: {remote_tcp_port}\n\n");
                    }
                }

                // ignore outgoing errors: those are normal
                e@SwarmEvent::OutgoingConnectionError { .. } => { log::debug!("Outgoing connection error: {e:?}"); }

                // otherwise log any other event
                e => { log::info!("Other event {e:?}"); }
            }
        }
    }
}
