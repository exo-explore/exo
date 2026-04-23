#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("babblerd is mac/linux-only");

pub mod babel;
pub mod config;
pub mod daemon;
pub mod dataplane;
pub mod fib;
pub mod identity;
pub(crate) mod route_ctl;
pub mod routing_stack;
pub mod tun;

pub use babel::babel;
pub use config::EXO_ULA_PREFIX as PREFIX;
pub use error::{BabbleError, Result};
pub use if_watcher::watch;
pub mod error {
    use std::io;
    use thiserror::Error;

    pub type Result<T> = core::result::Result<T, BabbleError>;

    #[derive(Error, Debug)]
    pub enum BabbleError {
        #[error("An IO error occurred: {0}")]
        Io(#[from] io::Error),
        #[error("Unspecified error")]
        Unspecified,
        #[error("Babeld crashed unexpectedly with code: {0:?}")]
        BabeldCrashed(Option<i32>),
        #[error("Failed to set IP address")]
        FailedToSetIp,
        #[error("Other error: {0}")]
        Other(String),
    }
}
pub mod if_watcher {
    #[cfg(target_os = "linux")]
    use std::path::PathBuf;
    use std::{collections::HashSet, net::IpAddr};

    use futures_lite::StreamExt;
    use n0_watcher::Watcher;
    use netwatch::interfaces::{Interface, IpNet};
    use tokio::sync::mpsc;

    use crate::config::{EXO_ULA_PREFIX, PHYSICAL_LINK_MTU, interface_allowlist_from_env};
    use crate::ip_manager::remove_ip;
    use crate::{BabbleError, Result, babel::Babble};

    pub const LOCALHOST_INTERFACE_NAMES: [&'static str; 2] = ["lo", "lo0"];

    trait IfaceExt {
        fn has_link_local_v6(&self) -> bool;
        fn has_required_mtu(&self) -> bool;
        fn is_real_interface(&self) -> bool;
        fn will_babel(&self) -> bool;
    }
    impl IfaceExt for Interface {
        fn will_babel(&self) -> bool {
            self.has_link_local_v6()
                && self.has_required_mtu()
                && self.is_real_interface()
                && self.is_up()
        }

        fn has_link_local_v6(&self) -> bool {
            let mut has = false;
            for addr in self.addrs() {
                let IpAddr::V6(a) = addr.addr() else {
                    continue;
                };
                if a.is_unicast_link_local() {
                    has = true;
                    break;
                }
            }
            has
        }

        fn has_required_mtu(&self) -> bool {
            let Some(mtu) = interface_mtu(self.name()) else {
                tracing::debug!(
                    "skipping interface {} because MTU could not be determined",
                    self.name()
                );
                return false;
            };
            if mtu < u32::from(PHYSICAL_LINK_MTU) {
                tracing::debug!(
                    "skipping interface {} because mtu {} is below required {}",
                    self.name(),
                    mtu,
                    PHYSICAL_LINK_MTU
                );
                return false;
            }
            true
        }

        fn is_real_interface(&self) -> bool {
            // macos is weird. en0 & en1 are ethernet & wifi (varies which is which by device). en3+ is thunderbolt, but at some point becomes usb ethernet.
            if self.name().strip_prefix("en").is_none()
            //.and_then(|s| s.parse::<u8>().ok())
            //.is_none_or(|_n| false)
            {
                return false;
            }
            #[cfg(target_os = "linux")]
            {
                if !PathBuf::from(format!("/sys/class/net/{}/device", self.name())).exists() {
                    tracing::debug!(
                        "skipping interface {} as it doesn't correspond to a physical link",
                        self.name()
                    );
                    return false;
                }
                let dev_type_path = PathBuf::from(format!("/sys/class/net/{}/type", self.name()));
                if !dev_type_path.exists() {
                    tracing::debug!(
                        "skipping interface {} with no type file at {:?}",
                        self.name(),
                        dev_type_path.to_str()
                    );
                    return false;
                }
                let Ok(dev_type) = std::fs::read_to_string(dev_type_path) else {
                    return false;
                };
                if dev_type.trim() != "1" {
                    tracing::debug!(
                        "skipping interface {} with type {:?}",
                        self.name(),
                        dev_type
                    );
                    return false;
                }
            }
            true
        }
    }

    fn interface_mtu(name: &str) -> Option<u32> {
        netdev::get_interfaces()
            .into_iter()
            .find(|iface| iface.name == name)
            .and_then(|iface| iface.mtu)
    }

    #[tracing::instrument(skip(send))]
    pub async fn watch(send: mpsc::Sender<Babble>) -> Result<()> {
        let mut ready_ifaces = HashSet::new();
        let interface_allowlist = interface_allowlist_from_env()
            .map_err(|e| BabbleError::Other(format!("invalid interface allowlist: {e}")))?;

        if let Some(allowlist) = &interface_allowlist {
            tracing::info!(?allowlist, "interface allowlist active");
        }

        tracing::info!("starting interface monitor");
        let mon = netwatch::netmon::Monitor::new()
            .await
            .map_err(|_| BabbleError::Unspecified)?;

        // TODD: this should never really be a thing thats the case, BUT I like the idea of having
        //       "heuristic" scripts that can help resolve issues but not necessarily gurantee success;
        //       I like the idea of generalising this concept into a framework where we have "heuristic tasks"
        //       that run to aid in tyring to fix some system ale-ment or whatever
        //
        // one-shot cleanup:
        // - remove any stale app-prefix addresses from lo0
        // - remove any app-prefix addresses that accidentally landed on physical links
        {
            let state = mon.interface_state();
            for iface in state.peek().interfaces.values() {
                let cleanup_target =
                    LOCALHOST_INTERFACE_NAMES.contains(&iface.name()) || iface.is_real_interface();
                if !cleanup_target {
                    continue;
                }
                for addr in iface.addrs() {
                    if let IpNet::V6 { net: v6, .. } = addr
                        && EXO_ULA_PREFIX.contains(&v6.addr())
                    {
                        tracing::info!("removing stale app ip {v6} from {}", iface.name());
                        if let Err(e) = remove_ip(v6, iface).await {
                            tracing::warn!(%e, "failed to remove stale app ip");
                        }
                    }
                }
            }
        }

        // stream updates
        let mut mon_stream = mon.interface_state().stream();
        while let Some(s) = mon_stream.next().await {
            for iface in s.interfaces.values() {
                if let Some(allowlist) = &interface_allowlist
                    && !allowlist.contains(iface.name())
                {
                    tracing::debug!(
                        "skipping interface {} because it is not in {}",
                        iface.name(),
                        crate::config::INTERFACE_ALLOWLIST_ENV
                    );
                    continue;
                }

                if !iface.is_real_interface() {
                    continue;
                }

                // physical links should not carry babbler application-space addresses
                for addr in iface.addrs() {
                    if let IpNet::V6 { net: v6, .. } = addr
                        && EXO_ULA_PREFIX.contains(&v6.addr())
                    {
                        tracing::info!("removing app ip {v6} from {}", iface.name());
                        if let Err(e) = remove_ip(v6, iface).await {
                            tracing::warn!(%e, "failed to remove ip");
                        }
                    }
                }
                if !iface.will_babel() {
                    continue;
                }
                if ready_ifaces.insert(iface.name().to_owned()) {
                    tracing::info!("telling babeld to watch {}", iface.name());
                    let Ok(()) = send.send(Babble::AddIface(iface.name().into())).await else {
                        return Ok(());
                    };
                }
            }
        }
        tracing::info!("stopping interface monitor");

        Ok(())
    }
}

pub(crate) mod ip_manager {
    pub use sys::add_ip;
    pub use sys::remove_ip;

    #[cfg(target_os = "linux")]
    mod sys {
        use ipnet::Ipv6Net;
        use netwatch::interfaces::Interface;

        use crate::{BabbleError, Result};
        use tokio::process::Command;

        #[tracing::instrument]
        pub async fn add_ip(subnet: Ipv6Net, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("add")
                .arg(format!("{subnet}"))
                .arg("dev")
                .arg(iface.name())
                .output()
                .await?;
            if out.status.success() {
                Ok(())
            } else {
                Err(BabbleError::FailedToSetIp)
            }
        }

        #[tracing::instrument]
        pub async fn remove_ip(v6: Ipv6Net, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("del")
                .arg(format!("{v6}"))
                .arg("dev")
                .arg(iface.name())
                .output()
                .await?;
            if out.status.success() {
                Ok(())
            } else {
                let std_err = String::from_utf8_lossy(&out.stdout);
                tracing::debug!(%std_err);
                Err(BabbleError::FailedToSetIp)
            }
        }
    }

    #[cfg(target_os = "macos")]
    mod sys {
        use ipnet::Ipv6Net;
        use netwatch::interfaces::Interface;

        use crate::BabbleError;
        use crate::Result;
        use tokio::process::Command;

        #[tracing::instrument]
        pub async fn add_ip(subnet: Ipv6Net, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{subnet}"))
                .arg("add")
                .output()
                .await?;
            if out.status.success() {
                Ok(())
            } else {
                Err(BabbleError::FailedToSetIp)
            }
        }

        #[tracing::instrument]
        pub async fn remove_ip(v6: Ipv6Net, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{v6}"))
                .arg("delete")
                .output()
                .await?;
            if out.status.success() {
                Ok(())
            } else {
                let std_err = String::from_utf8_lossy(&out.stdout);
                tracing::debug!(%std_err);
                Err(BabbleError::FailedToSetIp)
            }
        }
    }
}
