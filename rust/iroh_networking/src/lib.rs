use std::collections::BTreeSet;

use iroh::{
    Endpoint, EndpointId, SecretKey, TransportAddr,
    discovery::{
        Discovery, EndpointData, IntoDiscoveryError,
        mdns::{DiscoveryEvent, MdnsDiscovery},
    },
    endpoint::BindError,
    endpoint_info::EndpointIdExt,
    protocol::Router,
};
use iroh_gossip::{
    Gossip, TopicId,
    api::{ApiError, GossipReceiver, GossipSender},
};

use n0_error::{e, stack_error};
use n0_future::{Stream, StreamExt};
use tokio::sync::Mutex;

#[stack_error(derive, add_meta, from_sources)]
pub enum Error {
    #[error(transparent)]
    FailedBinding { source: BindError },
    /// The gossip topic was closed.
    #[error(transparent)]
    FailedCommunication { source: ApiError },
    #[error("No IP Protocol supported on device")]
    IPNotSupported { source: IntoDiscoveryError },
    #[error("No peers found before subscribing")]
    NoPeers,
}

#[derive(Debug)]
pub struct ExoNet {
    pub alpn: String,
    pub router: Router,
    pub gossip: Gossip,
    pub mdns: MdnsDiscovery,
    pub known_peers: Mutex<BTreeSet<EndpointId>>,
}

impl ExoNet {
    pub async fn init_iroh(sk: SecretKey, namespace: &str) -> Result<Self, Error> {
        let endpoint = Endpoint::empty_builder(iroh::RelayMode::Disabled)
            .secret_key(sk)
            .bind()
            .await?;
        let mdns = MdnsDiscovery::builder().build(endpoint.id())?;
        let endpoint_addr = endpoint.addr();

        let bound = endpoint_addr
            .ip_addrs()
            .map(|it| TransportAddr::Ip(it.clone()));

        log::info!("publishing {endpoint_addr:?} with mdns");
        mdns.publish(&EndpointData::new(bound));
        endpoint.discovery().add(mdns.clone());
        let alpn = format!("/exo_discovery_network/{}", namespace).to_owned();
        // max msg size 4MB
        let gossip = Gossip::builder()
            .max_message_size(4 * 1024 * 1024)
            .alpn(&alpn)
            .spawn(endpoint.clone());
        let router = Router::builder(endpoint)
            .accept(&alpn, gossip.clone())
            .spawn();
        Ok(Self {
            alpn,
            router,
            gossip,
            mdns,
            known_peers: Mutex::new(BTreeSet::new()),
        })
    }

    pub async fn start_auto_dialer(&self) {
        let mut recv = self.connection_info().await;

        log::info!(
            "Starting auto dialer for id {}",
            self.router.endpoint().id().to_z32()
        );
        while let Some(item) = recv.next().await {
            match item {
                DiscoveryEvent::Discovered { endpoint_info, .. } => {
                    let id = endpoint_info.endpoint_id;
                    if id == self.router.endpoint().id() {
                        continue;
                    }
                    if !self
                        .known_peers
                        .lock()
                        .await
                        .contains(&endpoint_info.endpoint_id)
                        && let Ok(conn) = self
                            .router
                            .endpoint()
                            .connect(endpoint_info, self.alpn.as_bytes())
                            .await
                        && conn.alpn() == self.alpn.as_bytes()
                    {
                        self.known_peers.lock().await.insert(id);
                        match self.gossip.handle_connection(conn).await {
                            Ok(()) => log::info!("Successfully dialled"),
                            Err(_) => log::info!("Failed to dial peer"),
                        }
                    }
                }
                DiscoveryEvent::Expired { endpoint_id } => {
                    log::info!("Peer expired {}", endpoint_id.to_z32());
                    self.known_peers.lock().await.remove(&endpoint_id);
                }
            }
        }
        log::info!("Auto dialer stopping");
    }

    pub async fn connection_info(&self) -> impl Stream<Item = DiscoveryEvent> + Unpin + use<> {
        self.mdns.subscribe().await
    }

    pub async fn subscribe(&self, topic: &str) -> Result<(GossipSender, GossipReceiver), Error> {
        if self.known_peers.lock().await.is_empty() {
            return Err(e!(Error::NoPeers));
        }
        Ok(self
            .gossip
            .subscribe_and_join(
                str_to_topic_id(topic),
                self.known_peers.lock().await.clone().into_iter().collect(),
            )
            .await?
            .split())
    }

    pub async fn shutdown(&self) {
        self.router
            .shutdown()
            .await
            .expect("Iroh Router failed to shutdown");
    }
}

fn str_to_topic_id(data: &str) -> TopicId {
    TopicId::from_bytes(*blake3::hash(data.as_bytes()).as_bytes())
}
