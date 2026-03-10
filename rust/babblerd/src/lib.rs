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
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }
    impl std::error::Error for BabbleError {}
    impl From<io::Error> for BabbleError {
        fn from(value: io::Error) -> Self {
            Self::Io(value)
        }
    }
}
pub mod if_watcher {
    use std::{collections::HashSet, net::Ipv6Addr, path::PathBuf};

    use futures_lite::StreamExt;
    use ipnet::Ipv6Net;
    use n0_watcher::Watcher;
    use netwatch::interfaces::IpNet;
    use tokio::sync::mpsc;

    use crate::{
        babel::Babble,
        ip_manager::add_ip,
        {BabbleError, Result},
    };

    pub const PREFIX: Ipv6Net = Ipv6Net::new_assert(
        Ipv6Addr::new(0xfde0, 0x20c6, 0x1fa7, 0xffff, 0, 0, 0, 0),
        64,
    );

    #[tracing::instrument(skip(send))]
    pub async fn watch(send: mpsc::Sender<Babble>) -> Result<()> {
        let mut ready_ifaces = HashSet::new();

        tracing::info!("starting interface monitor");
        let mon = netwatch::netmon::Monitor::new()
            .await
            .map_err(|_| BabbleError::Unspecified)?;
        let mut mon_stream = mon.interface_state().stream();

        while let Some(s) = mon_stream.next().await {
            let mut not_seen = ready_ifaces.clone();
            for iface in s.interfaces.values().filter(|iface| iface.is_up()) {
                if not_seen.contains(iface.name()) {
                    assert!(not_seen.remove(iface.name()));
                    continue;
                }
                #[cfg(target_os = "macos")]
                if !iface.name().starts_with("en") {
                    tracing::debug!("skipping non 'en' interface {}", iface.name());
                    continue;
                }
                #[cfg(target_os = "linux")]
                {
                    if !PathBuf::from(format!("/sys/class/net/{}/device", iface.name())).exists() {
                        tracing::debug!(
                            "skipping interface {} as it doesn't correspond to a physical link",
                            iface.name()
                        );
                        continue;
                    }
                    let dev_type_path =
                        PathBuf::from(format!("/sys/class/net/{}/type", iface.name()));
                    if !dev_type_path.exists() {
                        tracing::debug!(
                            "skipping interface {} with no type file at {:?}",
                            iface.name(),
                            dev_type_path.to_str()
                        );
                        continue;
                    }
                    let dev_type = tokio::fs::read_to_string(dev_type_path).await?;
                    if dev_type.trim() != "1" {
                        tracing::debug!(
                            "skipping interface {} with type {:?}",
                            iface.name(),
                            dev_type
                        );
                        continue;
                    }
                }
                let mut found_addr = false;
                for addr in iface.addrs() {
                    if let IpNet::V6(v6) = addr
                        && PREFIX.contains(&v6.addr())
                    {
                        found_addr = true;
                        break;
                    }
                }

                if !found_addr {
                    let lower: u64 = rand::random();
                    let upper = 0xfde0_20c6_1fa7_ffff_u64;
                    let addr = Ipv6Addr::from_bits((u128::from(upper) << 64) | u128::from(lower));
                    tracing::info!("adding new ip {} to {}", addr, iface.name());
                    add_ip(addr, iface)?;
                }

                tracing::info!("telling babeld to watch {}", iface.name());
                let Ok(()) = send.send(Babble::AddIface(iface.name().to_owned())).await else {
                    return Ok(());
                };
                ready_ifaces.insert(iface.name().to_owned());
            }
            for iface in not_seen {
                tracing::info!("telling babeld to stop watching {iface}");
                let Ok(()) = send.send(Babble::RemoveIface(iface.clone())).await else {
                    return Ok(());
                };
                assert!(ready_ifaces.remove(&iface));
            }
        }
        tracing::info!("stopping interface monitor");

        Ok(())
    }
}
pub mod babel {
    #[tracing::instrument(skip(sock, receiver))]
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
                        Err(_) => { tracing::debug!("receiver closed, dropping connection"); break; },
                        Ok(s) => write.write_all(format!("{s}\n").as_bytes()).await,
                    };
                    if let Err(e) = res { tracing::warn!(error=%e, "failed to write to socket"); break; };
                }
            };
        }
        tracing::info!("closing socket conn");
        _ = write.shutdown().await;
    }

    #[cfg(target_os = "macos")]
    const PRIVATE_SOCK_PATH: &str = "/var/run/babeld.sock";
    #[cfg(target_os = "linux")]
    const PRIVATE_SOCK_PATH: &str = "/run/babeld.sock";

    use std::io;
    use std::time::{Duration, Instant};

    use futures_lite::FutureExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
    use tokio::net::UnixStream;
    use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::process::Command;
    use tokio::sync::{broadcast, mpsc};

    use crate::{BabbleError, Result, if_watcher::PREFIX};

    #[derive(Debug)]
    pub enum Babble {
        AddIface(String),
        RemoveIface(String),
    }

    pub struct BabeldProcess {
        proc: tokio::process::Child,
    }
    impl Drop for BabeldProcess {
        fn drop(&mut self) {
            // emergency sigkill babeld process to prevent leakage
            if self.proc.try_wait().is_err() {
                _ = self.proc.start_kill();
            }
        }
    }
    impl BabeldProcess {
        #[tracing::instrument]
        fn spawn(first_iface: String) -> Result<Self> {
            match Command::new("babeld")
                .arg("-G")
                .arg(PRIVATE_SOCK_PATH)
                .arg("-C")
                .arg(format!("redistribute ip {PREFIX} local allow\n"))
                .arg("-C")
                .arg("redistribute local deny\n")
                .arg(first_iface)
                .spawn()
            {
                Ok(proc) => Ok(Self { proc }),
                Err(e) => {
                    tracing::warn!(error=%e, "failed to spawn babeld");
                    Err(e.into())
                }
            }
        }

        #[tracing::instrument(skip(read, write))]
        async fn query(
            read: &mut Lines<BufReader<OwnedReadHalf>>,
            write: &mut OwnedWriteHalf,
            cmd: &str,
        ) -> io::Result<Option<bool>> {
            write.write_all(cmd.as_bytes()).await?;
            loop {
                let Some(line) = read.next_line().await? else {
                    tracing::warn!("babeld closed unexpectedly");
                    return Ok(None);
                };
                tracing::info!("[babel] {:?}", line);
                if line == "ok" {
                    return Ok(Some(true));
                } else if line == "bad" {
                    tracing::warn!("malformed message sent to babeld");
                    return Ok(Some(false));
                }
            }
        }

        async fn supervise(
            &self,
            mut recv: mpsc::Receiver<Babble>,
            send: broadcast::Sender<String>,
        ) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            while !matches!(tokio::fs::try_exists(PRIVATE_SOCK_PATH).await, Ok(true)) {
                tracing::info!("where is the sock");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
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
            if Self::query(&mut babel_lines, &mut writer, "monitor\n")
                .await?
                .is_none()
            {
                return Ok(());
            };

            let mut time = Instant::now();
            loop {
                if Instant::now() - time > Duration::from_secs(5) {
                    if Self::query(&mut babel_lines, &mut writer, "dump\n")
                        .await?
                        .is_none()
                    {
                        return Ok(());
                    };
                    time = Instant::now();
                }
                tokio::select! {
                    babble = recv.recv() => {
                        tracing::debug!("[babble] {:?}", babble);
                        let Some(babble) = babble else {
                            break;
                        };
                        match babble {
                            Babble::AddIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, format!("interface {iface}\n").as_ref()).await?;
                            }
                            Babble::RemoveIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, format!("flush interface {iface}\n").as_ref()).await?;
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
            if let Some(pid) = self.proc.id() {
                let pid: libc::pid_t = pid.try_into().expect("pid overflow");
                // SAFETY: pid >= 0, freshly checked.
                let rc = unsafe { libc::kill(pid, libc::SIGINT) };
                if rc != 0 && rc != libc::ESRCH {
                    return Err(io::Error::last_os_error().into());
                }
                let exit_code = async { Some(self.proc.wait().await) }
                    .or(async {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        None
                    })
                    .await;
                match exit_code {
                    Some(Ok(code)) => {
                        if code.success() {
                            Ok(())
                        } else {
                            Err(BabbleError::BabeldCrashed(code.code()))
                        }
                    }
                    Some(Err(e)) => Err(e.into()),
                    None => {
                        self.proc.kill().await?;
                        Err(BabbleError::BabeldCrashed(None))
                    }
                }
            } else {
                Ok(())
            }
        }
    }

    #[tracing::instrument(skip(send, recv))]
    pub async fn babel(
        mut recv: mpsc::Receiver<Babble>,
        send: broadcast::Sender<String>,
    ) -> Result<()> {
        let first_iface = loop {
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

        let babel = BabeldProcess::spawn(first_iface)?;
        let ret = babel.supervise(recv, send).await;
        _ = babel.shutdown().await;
        ret
    }
}
pub(crate) mod ip_manager {
    pub(crate) use sys::add_ip;

    #[cfg(target_os = "linux")]
    mod sys {
        use netwatch::interfaces::Interface;

        use crate::{BabbleError, Result};
        use std::{net::Ipv6Addr, process::Command};

        #[tracing::instrument]
        pub fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("add")
                .arg(format!("{v6}/128"))
                .arg("dev")
                .arg(iface.name())
                .output()?;
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
        use std::{net::Ipv6Addr, process::Command};

        #[tracing::instrument]
        pub fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{v6}/128"))
                .arg("add")
                .output()?;
            if out.status.success() {
                Ok(())
            } else {
                Err(BabbleError::FailedToSetIp)
            }
        }
    }
}
