pub use error::{BabbleError, Result};
pub use if_watcher::watch;
pub mod error {
    use std::io;

    pub type Result<T> = core::result::Result<T, BabbleError>;
    #[derive(Debug)]
    pub enum BabbleError {
        Io(io::Error),
        Unspecified,
        BabeldCrashed(Option<i32>),
        FailedToSetIp,
        Other(String),
    }
    impl std::fmt::Display for BabbleError {
        // use the debug display for now
        #[inline]
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }
    impl std::error::Error for BabbleError {}
    impl From<io::Error> for BabbleError {
        #[inline]
        fn from(value: io::Error) -> Self {
            Self::Io(value)
        }
    }
}
pub mod if_watcher {
    #[cfg(target_os = "linux")]
    use std::path::PathBuf;
    use std::{
        collections::HashMap,
        net::{IpAddr, Ipv6Addr},
    };

    use futures_lite::StreamExt;
    use ipnet::Ipv6Net;
    use n0_watcher::Watcher;
    use netwatch::interfaces::{Interface, IpNet};
    use tokio::sync::mpsc;

    use crate::{
        BabbleError, Result,
        babel::Babble,
        ip_manager::{add_ip, remove_ip},
    };

    pub const PREFIX: Ipv6Net =
        Ipv6Net::new_assert(Ipv6Addr::new(0xfde0, 0x20c6, 0x1fa7, 0, 0, 0, 0, 0), 48);

    trait IfaceExt {
        fn has_link_local_v6(&self) -> bool;
        fn is_real_interface(&self) -> bool;
        fn will_babel(&self) -> bool;
        fn get_v6_in(&self, prefix: Ipv6Net) -> Option<Ipv6Addr>;
    }
    impl IfaceExt for Interface {
        fn will_babel(&self) -> bool {
            self.has_link_local_v6() && self.is_real_interface() && self.is_up()
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
        fn is_real_interface(&self) -> bool {
            #[cfg(target_os = "macos")]
            if !self.name().starts_with("en") {
                tracing::debug!("skipping non 'en' interface {}", self.name());
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
        fn get_v6_in(&self, prefix: Ipv6Net) -> Option<Ipv6Addr> {
            for addr in self.addrs() {
                if let IpNet::V6(v6) = addr
                    && prefix.contains(&v6.addr())
                {
                    return Some(v6.addr());
                }
            }
            None
        }
    }

    #[tracing::instrument(skip(send))]
    pub async fn watch(send: mpsc::Sender<Babble>) -> Result<()> {
        let mut ready_ifaces = HashMap::new();
        let ip_node_id = u128::from(rand::random::<u64>()) << 16;
        let my_range = Ipv6Net::new_assert(
            Ipv6Addr::from_bits(PREFIX.addr().to_bits() | ip_node_id),
            112,
        );
        // 0 is reserved for the first loopback address
        let mut iface_num: u16 = 1;

        tracing::info!("starting interface monitor");
        let mon = netwatch::netmon::Monitor::new()
            .await
            .map_err(|_| BabbleError::Unspecified)?;
        {
            let temp = mon.interface_state();
            let initial = &temp.peek().interfaces;
            for interface in ["lo", "lo0"] {
                if let Some(iface) = initial.get(interface) {
                    add_ip(my_range.addr(), iface).await?;
                    break;
                }
            }
        }
        let mut mon_stream = mon.interface_state().stream();

        while let Some(s) = mon_stream.next().await {
            let mut not_seen = ready_ifaces.clone();
            for iface in s.interfaces.values().filter(|iface| iface.will_babel()) {
                if not_seen.contains_key(iface.name()) {
                    assert!(not_seen.remove(iface.name()).is_some());
                    continue;
                }

                for addr in iface.addrs() {
                    if let IpNet::V6(v6) = addr
                        && PREFIX.contains(&v6)
                        && !my_range.contains(&v6)
                    {
                        tracing::info!("removing stale ip {v6} from {}", iface.name());
                        // don't really care if this fails
                        _ = remove_ip(v6.addr(), iface).await;
                    }
                }
                if iface.get_v6_in(my_range).is_none() {
                    let addr =
                        Ipv6Addr::from_bits(my_range.addr().to_bits() | u128::from(iface_num));
                    iface_num += 1;
                    assert!(iface_num < u16::MAX, "Really? u16::MAX interfaces?");
                    tracing::info!("adding new ip {addr} to {}", iface.name());
                    //add_ip(addr, iface).await?;
                }

                tracing::info!("telling babeld to watch {}", iface.name());
                let Ok(()) = send.send(Babble::AddIface(iface.name().to_owned())).await else {
                    return Ok(());
                };
                ready_ifaces.insert(iface.name().to_owned(), iface.clone());
            }
            for (name, iface) in not_seen {
                tracing::info!("telling babeld to stop watching {name}");
                if let Some(v6) = iface.get_v6_in(PREFIX) {
                    tracing::info!("removing stale ip {v6} from {name}");
                    // don't really care if this fails
                    _ = remove_ip(v6, &iface).await;
                }
                assert!(ready_ifaces.remove(&name).is_some());
                let Ok(()) = send.send(Babble::RemoveIface(name)).await else {
                    return Ok(());
                };
            }
        }
        tracing::info!("stopping interface monitor");

        Ok(())
    }
}
pub mod babel {
    #[tracing::instrument(skip_all)]
    pub async fn handle_listener(sock: UnixStream, mut receiver: broadcast::Receiver<String>) {
        tracing::info!("new socket conn");
        let (reader, mut write) = sock.into_split();
        let mut reader = BufReader::new(reader).lines();
        loop {
            tokio::select! {
                read = reader.next_line() => {
                    let Ok(Some(_)) = read else { break; };
                }
                recv = receiver.recv() => {
                    let res = match recv {
                        Err(broadcast::error::RecvError::Lagged(_)) => { tracing::warn!("receiver lagged"); continue; },
                        Err(broadcast::error::RecvError::Closed) => { tracing::debug!("receiver closed, dropping connection"); break; },
                        Ok(s) => write.write_all(format!("{s}\n").as_bytes()).await,
                    };
                    if let Err(e) = res { tracing::warn!(error=%e, "failed to write to socket"); break; }
                }
            };
        }
        tracing::info!("closing socket conn");
        _ = write.shutdown().await;
    }

    #[cfg(target_os = "macos")]
    const PRIVATE_SOCK_PATH: &str = "/var/run/babbler/private/babeld.sock";
    #[cfg(target_os = "linux")]
    const PRIVATE_SOCK_PATH: &str = "/run/babbler/private/babeld.sock";
    #[cfg(target_os = "macos")]
    const PRIVATE_DIR: &str = "/var/run/babbler/private";
    #[cfg(target_os = "linux")]
    const PRIVATE_DIR: &str = "/run/babbler/private";

    use std::fs::Permissions;
    use std::io;
    use std::os::unix::fs::PermissionsExt;
    use tokio::time::Duration;

    use futures_lite::FutureExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
    use tokio::net::UnixStream;
    use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::process::Command;
    use tokio::sync::{broadcast, mpsc};

    use crate::if_watcher::PREFIX;
    use crate::{BabbleError, Result};

    #[derive(Debug)]
    pub enum Babble {
        AddIface(String),
        RemoveIface(String),
    }

    pub struct BabeldProcess {
        proc: tokio::process::Child,
    }
    impl Drop for BabeldProcess {
        #[inline]
        fn drop(&mut self) {
            // emergency sigkill babeld process to prevent leakage
            if self.proc.try_wait().is_err() {
                _ = self.proc.start_kill();
            }
        }
    }
    impl BabeldProcess {
        #[tracing::instrument]
        async fn spawn(iface: String) -> Result<Self> {
            tokio::fs::create_dir_all(PRIVATE_DIR).await?;
            tokio::fs::set_permissions(PRIVATE_DIR, Permissions::from_mode(0o0600)).await?;
            tracing::info!("spawning babeld socket in {PRIVATE_SOCK_PATH}");
            let res = match Command::new("babeld")
                .arg("-G")
                .arg(PRIVATE_SOCK_PATH)
                .arg("-I")
                .arg(format!("{PRIVATE_DIR}/babeld.pid"))
                .arg("-C")
                .arg(format!("redistribute local ip {PREFIX}"))
                .arg("-C")
                .arg("redistribute local deny\n")
                .arg(iface)
                .spawn()
            {
                Ok(proc) => Ok(Self { proc }),
                Err(e) => {
                    tracing::warn!(error=%e, "failed to spawn babeld");
                    Err(e.into())
                }
            };
            tokio::time::sleep(Duration::from_millis(10)).await;
            while !matches!(tokio::fs::try_exists(PRIVATE_SOCK_PATH).await, Ok(true)) {
                tracing::info!("where is the sock");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            res
        }

        #[tracing::instrument(skip(read, write, send))]
        async fn query(
            read: &mut Lines<BufReader<OwnedReadHalf>>,
            write: &mut OwnedWriteHalf,
            send: &broadcast::Sender<String>,
            cmd: &str,
        ) -> io::Result<Option<bool>> {
            write.write_all(cmd.as_bytes()).await?;
            loop {
                let Some(line) = read.next_line().await? else {
                    tracing::warn!("babeld closed unexpectedly");
                    return Ok(None);
                };
                tracing::info!("[babel] {:?}", line);
                let ret = if line == "ok" {
                    Ok(Some(true))
                } else if line == "bad" {
                    tracing::warn!("malformed message sent to babeld");
                    Ok(Some(false))
                } else {
                    Ok(None)
                };
                let Ok(_) = send.send(line) else {
                    return Ok(None);
                };
                if !matches!(ret, Ok(None)) {
                    return ret;
                }
            }
        }

        #[tracing::instrument(skip_all)]
        async fn supervise(
            &self,
            mut recv: mpsc::Receiver<Babble>,
            send: broadcast::Sender<String>,
        ) -> Result<()> {
            std::fs::set_permissions(PRIVATE_SOCK_PATH, Permissions::from_mode(0o0600))?;
            tracing::debug!("connecting to babeld.sock");
            let (reader, mut writer) = UnixStream::connect(PRIVATE_SOCK_PATH).await?.into_split();
            let mut babel_lines = BufReader::new(reader).lines();
            while let Some(s) = babel_lines.next_line().await? {
                tracing::debug!("[babeld] {}", s);
                if s == "ok" {
                    break;
                }
            }
            tracing::info!("babeld ok");
            /* TODO(evan): push rather than pull
            if Self::query(&mut babel_lines, &mut writer, "monitor\n")
                .await?
                .is_none()
            {
                return Ok(());
            };
            */

            let mut interval = tokio::time::interval(Duration::from_secs(5));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if Self::query(&mut babel_lines, &mut writer, &send, "dump\n")
                            .await?
                            .is_none()
                        {
                            return Ok(());
                        }
                    },
                    babble = recv.recv() => {
                        tracing::debug!("[babble] {:?}", babble);
                        let Some(babble) = babble else {
                            break;
                        };
                        match babble {
                            Babble::AddIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, &send, format!("interface {iface}\n").as_ref()).await?;
                            }
                            Babble::RemoveIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, &send, format!("flush interface {iface}\n").as_ref()).await?;
                            }
                        }
                    },
                    line = babel_lines.next_line() => {
                        let Ok(Some(line)) = line else {
                            break;
                        };
                        tracing::debug!("[babeld] {}", line);
                        let Ok(_) = send.send(line) else {
                            break;
                        };
                    },
                }
            }
            Ok(())
        }

        async fn shutdown(mut self) -> Result<()> {
            let kill_res = if let Some(pid) = self.proc.id() {
                let pid: libc::pid_t = pid.try_into().expect("pid overflow");
                // SAFETY: pid >= 0, freshly checked.
                let rc = unsafe { libc::kill(pid, libc::SIGINT) };
                let rc_err = if rc != 0 && rc != libc::ESRCH {
                    Err(io::Error::last_os_error().into())
                } else {
                    Ok(())
                };
                let exit_code = async { Some(self.proc.wait().await) }
                    .or(async {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        None
                    })
                    .await;
                match exit_code {
                    Some(Ok(code)) => {
                        if code.success() {
                            rc_err
                        } else {
                            rc_err.and_then(|()| Err(BabbleError::BabeldCrashed(code.code())))
                        }
                    }
                    Some(Err(e)) => Err(e.into()),
                    None => {
                        self.proc.kill().await?;
                        rc_err.and(Err(BabbleError::BabeldCrashed(None)))
                    }
                }
            } else {
                Ok(())
            };
            let rem_res = match std::fs::remove_file(PRIVATE_SOCK_PATH) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e.into()),
            };
            kill_res.and(rem_res)
        }
    }

    #[tracing::instrument(skip(send, recv))]
    pub async fn babel(
        mut recv: mpsc::Receiver<Babble>,
        send: broadcast::Sender<String>,
    ) -> Result<()> {
        let iface = loop {
            match recv.recv().await {
                Some(Babble::AddIface(iface)) => {
                    break iface;
                }
                Some(Babble::RemoveIface(iface)) => {
                    tracing::warn!("ignored erroneous Babble::RemoveIface for {iface}");
                }
                None => return Ok(()),
            }
        };

        let babel = BabeldProcess::spawn(iface).await?;
        let res1 = babel.supervise(recv, send).await;
        let res2 = babel.shutdown().await;
        res1.and(res2)
    }
}
pub(crate) mod ip_manager {
    pub use sys::add_ip;
    pub use sys::remove_ip;

    #[cfg(target_os = "linux")]
    mod sys {
        use netwatch::interfaces::Interface;

        use crate::{BabbleError, Result};
        use std::net::Ipv6Addr;
        use tokio::process::Command;

        #[tracing::instrument]
        pub async fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("add")
                .arg(format!("{v6}/128"))
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
        pub async fn remove_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("del")
                .arg(format!("{v6}/128"))
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
    }

    #[cfg(target_os = "macos")]
    mod sys {
        use netwatch::interfaces::Interface;

        use crate::BabbleError;
        use crate::Result;
        use std::net::Ipv6Addr;
        use tokio::process::Command;

        #[tracing::instrument]
        pub async fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{v6}/128"))
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
        pub async fn remove_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{v6}/128"))
                .arg("delete")
                .output()
                .await?;
            if out.status.success() {
                Ok(())
            } else {
                Err(BabbleError::FailedToSetIp)
            }
        }
    }
}
