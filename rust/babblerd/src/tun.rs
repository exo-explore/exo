use ipnet::Ipv6Net;
use std::net::Ipv6Addr;
use std::sync::Arc;
use tun_rs::{DeviceBuilder, SyncDevice};

use crate::config::TUN_MTU;

#[cfg(target_os = "linux")]
const DESIRED_TUN_NAME: &str = "exonet";

/// Holds the TUN device open for the lifetime of the daemon.
/// The interface disappears when this is dropped.
pub struct TunDevice {
    dev: Arc<SyncDevice>,
    ifname: String,
    node_addr: Ipv6Net, // TODO: we are only ever gonna install /128 subnets, maybe change to Ipv6Addr in future??
}

impl TunDevice {
    pub fn create(node_addr: Ipv6Addr) -> crate::Result<Self> {
        let builder = DeviceBuilder::new().ipv6(node_addr, 128u8).mtu(TUN_MTU);
        #[cfg(target_os = "linux")]
        let builder = builder.name(DESIRED_TUN_NAME);

        let dev = builder
            .with(|builder| {
                builder.packet_information(false);

                #[cfg(target_os = "macos")]
                {
                    // Route ownership stays in userspace; do not let tun-rs auto-add routes.
                    // The IPv6 /128 address itself is still applied by tun-rs.
                    builder.associate_route(false);
                }
            })
            .build_sync()?;

        dev.set_nonblocking(true)?;
        let ifname = dev.name()?;

        Ok(Self {
            dev: Arc::new(dev),
            ifname,
            node_addr: Ipv6Net::new_assert(node_addr, 128), // TODO: i dont't like the magic numbers, I also don't like wrapping and unwrapping
        })
    }

    pub fn ifname(&self) -> &str {
        &self.ifname
    }

    pub fn node_addr(&self) -> Ipv6Net {
        self.node_addr
    }

    pub fn device(&self) -> &SyncDevice {
        self.dev.as_ref()
    }

    pub fn shared_device(&self) -> Arc<SyncDevice> {
        Arc::clone(&self.dev)
    }
}
