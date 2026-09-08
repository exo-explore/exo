use std::{
    io,
    net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6},
    sync::Arc,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};
use log::{debug, trace, warn};
use netwatcher::WatchHandle;
use parking_lot::Mutex;
use tokio::{
    net::UdpSocket,
    time::{Interval, interval},
};
use zenoh::config::ZenohId;

use crate::mdns::MdnsDiscovery;
use crate::peers::iface_is_wired;

const GROUP: Ipv6Addr = Ipv6Addr::new(0xff12, 0, 0, 0, 0, 0, 0xe0a1, 0xde89);
const GROUP_V4: Ipv4Addr = Ipv4Addr::new(239, 255, 224, 161);
const MAGIC: [u8; 3] = *b"EXO";

/// `netwatcher::WatchHandle` wraps a raw OS handle (`*mut c_void` on Windows)
/// which is not automatically `Send`/`Sync`. The handle is a kernel object and
/// is only stored so interface-change callbacks keep running.
struct SendWatchHandle(#[allow(dead_code)] WatchHandle);

// SAFETY: the wrapped handle is an OS watcher resource. It is not accessed
// concurrently (we never lock or call into it after construction; Drop is the
// only use) and Windows HANDLEs / Unix watcher FDs may be released from any
// thread.
unsafe impl Send for SendWatchHandle {}
unsafe impl Sync for SendWatchHandle {}

pub struct Discovery {
    sock: Option<Arc<UdpSocket>>,
    sock_v4: Option<Arc<UdpSocket>>,
    ifaces: Arc<Mutex<Vec<SocketAddrV6>>>,
    v4_ifaces: Arc<Mutex<Vec<Ipv4Addr>>>,
    ethernet_v4: Arc<Mutex<Vec<Ipv4Addr>>>,
    namespace: [u8; 8],
    last_nonce: Mutex<[u8; 8]>,
    /// the port of the service we are doing discovery for - transmitted to peers
    listen_port: u16,
    discovery_port: u16,
    zid: ZenohId,
    tick: Interval,
    mdns: Option<MdnsDiscovery>,
    _sync: SendWatchHandle,
}

#[derive(Debug, Clone, Copy)]
pub struct Discovered {
    pub zid: ZenohId,
    pub addr: SocketAddr,
}

impl Discovery {
    pub async fn new(
        zid: ZenohId,
        namespace: [u8; 8],
        listen_port: u16,
        discovery_port: u16,
    ) -> io::Result<Self> {
        let sock = match bind_v6_discovery(discovery_port) {
            Ok(v6) => Some(Arc::new(v6)),
            Err(e) => {
                warn!("IPv6 discovery socket unavailable: {e}");
                None
            }
        };

        let sock_v4 = match bind_v4_discovery(discovery_port) {
            Ok(v4) => Some(Arc::new(v4)),
            Err(e) => {
                warn!("IPv4 discovery socket unavailable: {e}");
                None
            }
        };

        if sock.is_none() && sock_v4.is_none() {
            return Err(io::Error::other(
                "failed to bind IPv4 and IPv6 discovery sockets",
            ));
        }

        let ifaces: Arc<Mutex<Vec<SocketAddrV6>>> = Default::default();
        let v4_ifaces: Arc<Mutex<Vec<Ipv4Addr>>> = Default::default();
        let ethernet_v4: Arc<Mutex<Vec<Ipv4Addr>>> = Default::default();
        let _sync = SendWatchHandle(
            netwatcher::watch_interfaces_with_callback({
                let sock = sock.clone();
                let sock_v4 = sock_v4.clone();
                let ifaces = ifaces.clone();
                let v4_ifaces = v4_ifaces.clone();
                let ethernet_v4 = ethernet_v4.clone();
                move |update| {
                    for (iface_idx, iface) in update.interfaces.iter() {
                        if iface
                            .ipv6_ips()
                            .all(|addr| addr.is_loopback() || addr.is_unspecified())
                        {
                            continue;
                        }

                        let Some(sock) = sock.as_ref() else {
                            continue;
                        };
                        match sock.join_multicast_v6(&GROUP, *iface_idx) {
                            Ok(()) => ifaces.lock().push(SocketAddrV6::new(
                                GROUP,
                                discovery_port,
                                0,
                                *iface_idx,
                            )),
                            Err(e) if e.kind() != io::ErrorKind::AddrInUse => {
                                // skip AddrInUse - just means we've already joined the mv6
                                if let Some(iface) = update.interfaces.get(&iface_idx) {
                                    warn!(
                                        "failed to join multicast v6 for interface {}: {e}",
                                        iface.name
                                    )
                                }
                            }
                            _ => {}
                        }
                    }
                    for iface_idx in update.diff.removed {
                        ifaces.lock().retain(|addr| addr.scope_id() != iface_idx);

                        let Some(sock) = sock.as_ref() else {
                            continue;
                        };
                        if let Err(e) = sock.leave_multicast_v6(&GROUP, iface_idx) {
                            if let Some(iface) = update.interfaces.get(&iface_idx) {
                                warn!(
                                    "failed to leave multicast v6 for interface {}: {e}",
                                    iface.name
                                )
                            }
                        }
                    }

                    let mut next_v4: Vec<Ipv4Addr> = Vec::new();
                    let mut next_eth: Vec<Ipv4Addr> = Vec::new();
                    for iface in update.interfaces.values() {
                        let wired = iface_is_wired(&iface.name);
                        for ip in iface.ipv4_ips().copied() {
                            if ip.is_loopback() || ip.is_unspecified() {
                                continue;
                            }
                            if let Some(v4) = sock_v4.as_ref() {
                                match v4.join_multicast_v4(GROUP_V4, ip) {
                                    Ok(()) => {}
                                    Err(e) if e.kind() == io::ErrorKind::AddrInUse => {}
                                    Err(e) => warn!(
                                        "failed to join IPv4 multicast on {} ({ip}): {e}",
                                        iface.name
                                    ),
                                }
                            }
                            next_v4.push(ip);
                            if wired {
                                next_eth.push(ip);
                            }
                        }
                    }
                    *v4_ifaces.lock() = next_v4;
                    *ethernet_v4.lock() = next_eth;
                }
            })
            // todo: better error handling here
            .expect("failed to bind discovery watcher"),
        );

        let mdns = if std::env::var("EXO_DISABLE_MDNS").ok().as_deref() == Some("1") {
            None
        } else {
            match MdnsDiscovery::start(zid, namespace, listen_port) {
                Ok(mdns) => {
                    debug!("mDNS discovery advertised as {}", crate::mdns::SERVICE_TYPE);
                    Some(mdns)
                }
                Err(e) => {
                    warn!("mDNS unavailable ({e}); IPv4/IPv6 UDP discovery still active");
                    None
                }
            }
        };

        Ok(Self {
            sock,
            sock_v4,
            namespace,
            ifaces,
            v4_ifaces,
            ethernet_v4,
            last_nonce: Mutex::new(rand::random()),
            listen_port,
            discovery_port,
            zid,
            tick: interval(Duration::from_secs(1)),
            mdns,
            _sync,
        })
    }

    pub fn ethernet_ipv4(&self) -> Vec<Ipv4Addr> {
        self.ethernet_v4.lock().clone()
    }

    pub async fn next(&mut self) -> io::Result<Discovered> {
        let mut buf_v6 = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        let mut buf_v4 = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        loop {
            tokio::select! {
                _ = self.tick.tick() => {
                    self.announce().await?;
                }
                res = recv_from_opt(&self.sock, &mut buf_v6) => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    if let Some(discovered) = self.respond(bytes_read, addr, &buf_v6).await? {
                        return Ok(discovered)
                    }
                }
                res = recv_from_opt(&self.sock_v4, &mut buf_v4) => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    if let Some(discovered) = self.respond(bytes_read, addr, &buf_v4).await? {
                        return Ok(discovered)
                    }
                }
                maybe = recv_mdns(&mut self.mdns) => {
                    if let Some(discovered) = maybe {
                        return Ok(discovered);
                    }
                }
            }
        }
    }

    async fn respond(
        &self,
        bytes_read: usize,
        addr: SocketAddr,
        buf: &[u8],
    ) -> io::Result<Option<Discovered>> {
        trace!(
            "raw recv: {bytes_read} bytes from {addr}: {:02x?}",
            &buf[..bytes_read]
        );
        if bytes_read < size_of::<Header>() {
            trace!("dropped: early EOF");
            return Ok(None);
        }
        let header: &Header = bytemuck::from_bytes(&buf[0..size_of::<Header>()]);
        if header.magic != MAGIC {
            trace!("dropped: wrong magic");
            return Ok(None);
        }
        let Ok(kind) = header.kind.try_into() else {
            trace!("dropped: unknown message kind {}", header.kind);
            return Ok(None);
        };
        match kind {
            Kind::Hello => {
                let total = Hello::buf_size();
                if bytes_read != total {
                    trace!("dropped: hello wrong size");
                    return Ok(None);
                }
                let hello: &Hello = bytemuck::from_bytes(&buf[size_of::<Header>()..total]);
                if hello.nonce == *self.last_nonce.lock() {
                    trace!("dropped: local hello nonce");
                    return Ok(None);
                }
                if hello.namespace != self.namespace {
                    trace!("dropped: different namespace");
                    return Ok(None);
                }

                // reply
                trace!("replying to Hello({:?})", hello.nonce);
                let reply = WhatsUp {
                    nonce: hello.nonce,
                    zid: self.zid.to_le_bytes(),
                    port_le: self.listen_port.to_le_bytes(),
                }
                .alloc();

                let reply_sock = match addr {
                    SocketAddr::V4(_) => self.sock_v4.as_deref().or(self.sock.as_deref()),
                    SocketAddr::V6(_) => self.sock.as_deref().or(self.sock_v4.as_deref()),
                };
                let Some(reply_sock) = reply_sock else {
                    return Ok(None);
                };

                for i in 1..6 {
                    if reply_sock
                        .send_to(&reply, addr)
                        .await
                        .inspect_err(|e| debug!("send to {addr} failed: {e}"))
                        .is_ok_and(|sent| sent == WhatsUp::buf_size())
                    {
                        trace!(
                            "sent {} bytes to {addr} after {} attempt(s)",
                            WhatsUp::buf_size(),
                            i
                        );
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
                Ok(None)
            }
            Kind::WhatsUp => {
                let total = WhatsUp::buf_size();
                if bytes_read != total {
                    trace!("dropped: whatsup wrong size");
                    return Ok(None);
                }
                let whats_up: &WhatsUp = bytemuck::from_bytes(&buf[size_of::<Header>()..total]);
                if whats_up.nonce != *self.last_nonce.lock() {
                    trace!("dropped: stale nonce");
                    return Ok(None);
                }
                let Ok(zid) = ZenohId::try_from(&whats_up.zid[..]) else {
                    trace!("dropped: zenoh conversion failed");
                    return Ok(None);
                };
                if zid == self.zid {
                    trace!("dropped: self zenoh id");
                    return Ok(None);
                }
                let port = u16::from_le_bytes(whats_up.port_le);
                let mut peer = addr;
                peer.set_port(port);
                Ok(Some(Discovered { addr: peer, zid }))
            }
        }
    }

    async fn announce(&self) -> io::Result<()> {
        let nonce = rand::random();
        *self.last_nonce.lock() = nonce;
        let buf = Hello {
            nonce,
            namespace: self.namespace,
        }
        .alloc();

        if let Some(sock) = &self.sock {
            let addrs = self.ifaces.lock().clone();
            debug!("announcing Hello({nonce:?}) to {addrs:?}");
            // rev so .remove() doesn't break things
            for (i, addr) in addrs.into_iter().enumerate().rev() {
                match sock.send_to(&buf, addr).await {
                    Ok(bytes) => trace!("sent {bytes} to {addr}"),
                    Err(e) if e.kind() == io::ErrorKind::HostUnreachable => {
                        debug!("disabling discovery address {addr}: {e}");
                        _ = self.ifaces.lock().swap_remove(i);
                    }
                    Err(e) => debug!("failed to reach {addr}: {e}"),
                }
            }
        }

        self.announce_v4(&buf).await;
        Ok(())
    }

    async fn announce_v4(&self, buf: &[u8]) {
        let Some(sock) = &self.sock_v4 else {
            return;
        };
        let multicast = SocketAddr::from((GROUP_V4, self.discovery_port));
        let limited_broadcast = SocketAddr::from((Ipv4Addr::BROADCAST, self.discovery_port));
        for dest in [multicast, limited_broadcast] {
            if let Err(e) = sock.send_to(buf, dest).await {
                debug!("IPv4 announce to {dest} failed: {e}");
            }
        }

        let locals = self.v4_ifaces.lock().clone();
        for ip in locals {
            match std::net::UdpSocket::bind((ip, 0)) {
                Ok(bound) => {
                    if let Err(e) = bound.set_broadcast(true) {
                        debug!("broadcast flag on {ip}: {e}");
                        continue;
                    }
                    if let Err(e) = bound.send_to(buf, limited_broadcast) {
                        debug!("iface broadcast from {ip}: {e}");
                    }
                    if let Err(e) = bound.send_to(buf, multicast) {
                        debug!("iface multicast from {ip}: {e}");
                    }
                }
                Err(e) => debug!("bind announce socket on {ip}: {e}"),
            }
        }
    }
}

fn bind_v6_discovery(discovery_port: u16) -> io::Result<UdpSocket> {
    let sock = socket2::Socket::new(
        socket2::Domain::IPV6,
        socket2::Type::DGRAM,
        Some(socket2::Protocol::UDP),
    )?;
    sock.set_reuse_address(true)?;
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, discovery_port, 0, 0).into())?;
    sock.set_nonblocking(true)?;
    sock.set_multicast_loop_v6(true)?;
    UdpSocket::from_std(sock.into())
}

fn bind_v4_discovery(discovery_port: u16) -> io::Result<UdpSocket> {
    let sock = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::DGRAM,
        Some(socket2::Protocol::UDP),
    )?;
    sock.set_reuse_address(true)?;
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.set_broadcast(true)?;
    sock.bind(&SocketAddr::from((Ipv4Addr::UNSPECIFIED, discovery_port)).into())?;
    sock.set_nonblocking(true)?;
    sock.set_multicast_loop_v4(true)?;
    sock.set_multicast_ttl_v4(1)?;
    UdpSocket::from_std(sock.into())
}

async fn recv_from_opt(
    sock: &Option<Arc<UdpSocket>>,
    buf: &mut [u8],
) -> io::Result<(usize, SocketAddr)> {
    match sock {
        Some(s) => s.recv_from(buf).await,
        None => std::future::pending().await,
    }
}

async fn recv_mdns(mdns: &mut Option<MdnsDiscovery>) -> Option<Discovered> {
    match mdns {
        Some(mdns) => mdns.rx.recv().await,
        None => std::future::pending().await,
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
// packet & version
pub enum Kind {
    Hello = 0,
    WhatsUp = 1,
}

pub struct UnknownKind;
impl TryFrom<u8> for Kind {
    type Error = UnknownKind;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Hello),
            1 => Ok(Self::WhatsUp),
            _ => Err(UnknownKind),
        }
    }
}

pub trait Message: Pod {
    const KIND: Kind;
}
// should be part of the Message trait, but const in traits isnt stabilized. this lets alloc :: Self -> [u8; Self::buf_size()]
macro_rules! impl_alloc {
    ($a:ident) => {
        impl $a {
            const fn buf_size() -> usize {
                size_of::<Header>() + size_of::<Self>()
            }
            pub fn alloc(self) -> [u8; Self::buf_size()] {
                let mut buf = [0u8; Self::buf_size()];
                buf[0..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&Header {
                    magic: MAGIC,
                    kind: Self::KIND as u8,
                }));
                buf[size_of::<Header>()..Self::buf_size()]
                    .copy_from_slice(bytemuck::bytes_of(&self));
                buf
            }
        }
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Header {
    magic: [u8; 3],
    kind: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Hello {
    pub nonce: [u8; 8],
    pub namespace: [u8; 8],
}
impl Message for Hello {
    const KIND: Kind = Kind::Hello;
}
impl_alloc!(Hello);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhatsUp {
    pub nonce: [u8; 8],
    pub zid: [u8; 16],
    pub port_le: [u8; 2],
}
impl Message for WhatsUp {
    const KIND: Kind = Kind::WhatsUp;
}
impl_alloc!(WhatsUp);
