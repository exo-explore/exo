pub use error::{Error, Result};
pub use if_watcher::watch;
pub mod error {
    use std::io;

    pub type Result<T> = core::result::Result<T, Error>;
    #[derive(Debug)]
    pub enum Error {
        Io(io::Error),
        Unspecified,
        BabeldCrashed(Option<i32>),
        FailedToSetIp,
        Other(String),
    }
    impl std::fmt::Display for Error {
        // use the debug display for now
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }
    impl std::error::Error for Error {}
    impl From<io::Error> for Error {
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
        {Error, Result},
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
            .map_err(|_| Error::Unspecified)?;
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
                let Ok(()) = send.send(Babble::AddIface(iface.name().to_string())).await else {
                    return Ok(());
                };
                ready_ifaces.insert(iface.name().to_string());
            }
            for iface in not_seen {
                tracing::info!("telling babeld to stop watching {iface}");
                let Ok(()) = send.send(Babble::RemoveIface(iface.to_string())).await else {
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
    pub async fn handle_listener(
        sock: UnixStream,
        mut receiver: broadcast::Receiver<String>,
    ) -> std::io::Result<()> {
        tracing::info!("new socket conn");
        let (read, mut write) = sock.into_split();
        let mut read = BufReader::new(read).lines();
        let res = loop {
            tokio::select! {
                read = read.next_line() => {
                    if let None = read? {
                        break Ok(());
                    }
                }
                recv = receiver.recv() => {
                    if let Ok(s) = recv {
                        if let Err(e) = write.write_all(format!("{s}\n").as_bytes()).await {
                            break Err(e)
                        };
                        continue;
                    }
                }
            };
        };
        tracing::info!("closing socket conn");
        write.shutdown().await.and(res)
    }

    #[cfg(target_os = "macos")]
    const PRIVATE_SOCK_PATH: &str = "/var/run/babeld.sock";
    #[cfg(target_os = "linux")]
    const PRIVATE_SOCK_PATH: &str = "/run/babeld.sock";

    use std::io;
    use std::time::Duration;

    use futures_lite::FutureExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
    use tokio::net::UnixStream;
    use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::process::Command;
    use tokio::sync::{broadcast, mpsc};

    use crate::{Error, Result, if_watcher::PREFIX};

    #[derive(Debug)]
    pub enum Babble {
        AddIface(String),
        RemoveIface(String),
    }

    pub struct BabeldProcess {
        proc: tokio::process::Child,
    }
    impl BabeldProcess {
        #[tracing::instrument]
        fn spawn(first_iface: String) -> Result<Self> {
            match Command::new("babeld")
                .arg("-G")
                .arg(PRIVATE_SOCK_PATH)
                .arg("-C")
                .arg(format!("redistribute ip {} local allow\n", PREFIX))
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
                    return Ok(None);
                };
                tracing::info!("[babel] {}", line);
                if line == "ok" {
                    return Ok(Some(true));
                } else if line == "bad" {
                    return Ok(Some(false));
                }
            }
        }

        async fn supervise(
            &mut self,
            mut recv: mpsc::Receiver<Babble>,
            send: broadcast::Sender<String>,
        ) -> Result<()> {
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;
                match tokio::fs::try_exists(PRIVATE_SOCK_PATH).await {
                    Ok(true) => break,
                    _ => {
                        tracing::info!("where is the sock");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
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
            Self::query(&mut babel_lines, &mut writer, "monitor\n").await?;

            loop {
                tokio::select! {
                    babble = recv.recv() => {
                        tracing::debug!("[babble] {:?}", babble);
                        let Some(babble) = babble else {
                            break;
                        };
                        match babble {
                            Babble::AddIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, format!("interface {}\n", iface).as_ref()).await?;
                            }
                            Babble::RemoveIface(iface) => {
                                Self::query(&mut babel_lines, &mut writer, format!("flush interface {}\n", iface).as_ref()).await?;
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
                let pid = pid as libc::pid_t;
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
                    Some(exit_code) => {
                        let code = exit_code?;
                        if code.success() {
                            Ok(())
                        } else {
                            Err(Error::BabeldCrashed(code.code()))
                        }
                    }
                    None => {
                        self.proc.kill().await?;
                        Err(Error::BabeldCrashed(None))
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
                    tracing::warn!("ignored erroneous Babble::RemoveIface for {iface}")
                }
                None => return Ok(()),
            }
        };

        let mut babel = BabeldProcess::spawn(first_iface)?;
        let ret = babel.supervise(recv, send).await;
        babel.shutdown().await?;
        ret
    }
}
mod ip_manager {
    pub(crate) use sys::add_ip;

    #[cfg(target_os = "linux")]
    mod sys {
        use netwatch::interfaces::Interface;

        use crate::{Error, Result};
        use std::{net::Ipv6Addr, process::Command};

        #[tracing::instrument]
        pub(crate) fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ip")
                .arg("addr")
                .arg("add")
                .arg(format!("{}/128", v6))
                .arg("dev")
                .arg(iface.name())
                .output()?;
            if out.status.success() {
                Ok(())
            } else {
                Err(Error::FailedToSetIp)
            }
        }
    }

    #[cfg(target_os = "macos")]
    mod sys {
        use netwatch::interfaces::Interface;

        use crate::Error;
        use crate::Result;
        use std::{net::Ipv6Addr, process::Command};

        #[tracing::instrument]
        pub(crate) fn add_ip(v6: Ipv6Addr, iface: &Interface) -> Result<()> {
            let out = Command::new("ifconfig")
                .arg(iface.name())
                .arg("inet6")
                .arg(format!("{}/128", v6.to_string()))
                .arg("add")
                .output()?;
            if out.status.success() {
                Ok(())
            } else {
                Err(Error::FailedToSetIp)
            }
        }
    }
}
