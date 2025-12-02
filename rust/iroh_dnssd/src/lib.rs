use std::{net::SocketAddr, sync::Mutex};

use serde::{Serialize, Deserialize};
use astro_dnssd::{DNSServiceBuilder, RegisteredDnsService};
use iroh::{
    Endpoint, EndpointId,
    discovery::{Discovery, DiscoveryError, DiscoveryItem, EndpointData},
};
use n0_future::boxed::BoxStream;
use rand::Rng;

#[derive(Debug)]
pub struct DnssdDiscovery {
    id: EndpointId,
    current_service: Mutex<RegisteredDnsService>,
}

impl DnssdDiscovery {
    pub fn spawn(endpoint: Endpoint) -> Self {
        Self {
            id: endpoint.id(),
            current_service: Mutex::new(build_from(endpoint.id(), endpoint.addr().ip_addrs())),
        }
    }
}

impl Discovery for DnssdDiscovery {
    fn publish(&self, _data: &EndpointData) {
        *self.current_service.lock().expect("mutex poison") = build_from(self.id, _data.ip_addrs())
    }
    fn resolve(
        &self,
        _endpoint_id: EndpointId,
    ) -> Option<BoxStream<Result<DiscoveryItem, DiscoveryError>>> {
        None
    }
}

fn build_from<'a, I: Iterator<Item = &'a SocketAddr>>(id: EndpointId, addrs: I) -> RegisteredDnsService {
    DNSServiceBuilder::new("_exo._iroh._udp", rand::rng().random_range(49152..65535))
        .with_key_value(
            "payload".to_owned(),
            serde_json::to_string(&Payload {
                id,
                addrs: addrs.cloned().collect()

            }).expect("serialization failed"),
        )
        .register()
        .expect("mdns register failed")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Payload {
    addrs: Vec<SocketAddr>,
    id: EndpointId,
}
