use std::collections::BTreeSet;

use iroh::{
    Endpoint, SecretKey,
    discovery::{
        IntoDiscoveryError,
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

use n0_error::stack_error;
use n0_future::{Stream, StreamExt};

#[stack_error(derive, add_meta, from_sources)]
pub enum Error {
    #[error(transparent)]
    FailedBinding { source: BindError },
    /// The gossip topic was closed.
    #[error(transparent)]
    FailedCommunication { source: ApiError },
    #[error("No IP Protocol supported on device")]
    IPNotSupported { source: IntoDiscoveryError },
}

#[derive(Debug)]
pub struct ExoNet {
    alpn: String,
    router: Router,
    gossip: Gossip,
    mdns: MdnsDiscovery,
}

impl ExoNet {
    pub async fn init_iroh(sk: SecretKey, namespace: &str) -> Result<Self, Error> {
        let endpoint = Endpoint::empty_builder(iroh::RelayMode::Disabled)
            .secret_key(sk)
            .bind()
            .await?;
        let mdns = MdnsDiscovery::builder().build(endpoint.id())?;
        endpoint.discovery().add(mdns.clone());
        let alpn = format!("/exo_discovery_network/{}", namespace).to_owned();
        let gossip = Gossip::builder().alpn(&alpn).spawn(endpoint.clone());
        let router = Router::builder(endpoint)
            .accept(&alpn, gossip.clone())
            .spawn();
        Ok(Self {
            alpn,
            router,
            gossip,
            mdns,
        })
    }

    pub async fn start_auto_dialer(&self) {
        let mut dialed = BTreeSet::new();
        let mut recv = self.connection_info().await;
        while let Some(item) = recv.next().await {
            match item {
                DiscoveryEvent::Discovered { endpoint_info, .. } => {
                    if !dialed.contains(&endpoint_info.endpoint_id) {
                        log::info!("Dialing new peer {}", endpoint_info.endpoint_id.to_z32());
                        let _ = self
                            .router
                            .endpoint()
                            .connect(endpoint_info, self.alpn.as_bytes())
                            .await;
                    } else {
                        dialed.insert(endpoint_info.endpoint_id);
                    }
                }
                DiscoveryEvent::Expired { endpoint_id } => {
                    dialed.remove(&endpoint_id);
                }
            }
        }
    }

    pub async fn connection_info(&self) -> impl Stream<Item = DiscoveryEvent> + Unpin + use<> {
        self.mdns.subscribe().await
    }

    pub async fn subscribe(&self, topic: &str) -> Result<(GossipSender, GossipReceiver), Error> {
        Ok(self
            .gossip
            .subscribe(str_to_topic_id(topic), vec![])
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

// Dead code here is for asserting these compile
#[allow(dead_code)]
#[cfg(test)]
mod test {
    use iroh::{SecretKey, discovery::mdns::DiscoveryEvent};

    use crate::ExoNet;

    fn is_send<T: Send>(_: &T) {}

    trait Probe: Send {}
    impl Probe for ExoNet {}
    impl Probe for DiscoveryEvent {}

    #[test]
    fn test_is_send() {
        // todo: make rand a dev dep.
        let fut = ExoNet::init_iroh(SecretKey::generate(&mut rand::rng()), "");
        is_send(&fut);
    }
}
