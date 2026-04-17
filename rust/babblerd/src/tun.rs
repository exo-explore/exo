use std::net::Ipv6Addr;
use ipnet::Ipv6Net;
use tun_rs::{DeviceBuilder, SyncDevice};

/// Holds the utun device open for the lifetime of the daemon.
/// The interface disappears when this is dropped.
pub struct UtunDevice {
    dev: SyncDevice,
    ifname: String,
    advertised: Ipv6Net, // TODO: we are only ever gonna install /128 subnets, maybe change to Ipv6Addr in future??
}

impl UtunDevice {
    pub fn create(advertised: Ipv6Addr) -> crate::Result<Self> {
        let dev = DeviceBuilder::new()
            .ipv6(advertised, 128u8)
            .mtu(1500) // TODO: is this patform specific??? is this a magic number??
            .with(|builder| {
                #[cfg(target_os = "macos")]
                {
                    // Babel owns remote routing; do not let tun-rs auto-add routes.
                    builder.associate_route(false).packet_information(false);
                }
            })
            .build_sync()?;

        let ifname = dev.name()?;

        Ok(Self {
            dev,
            ifname,
            advertised: Ipv6Net::new_assert(advertised, 128), // TODO: i dont't like the magic numbers, I also don't like wrapping and unwrapping
        })
    }

    pub fn ifname(&self) -> &str {
        &self.ifname
    }

    pub fn advertised(&self) -> Ipv6Net {
        self.advertised
    }

    pub fn device(&self) -> &SyncDevice {
        &self.dev
    }
}
