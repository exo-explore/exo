use std::{
    io::{self, ErrorKind},
    net::{Ipv6Addr, SocketAddr, SocketAddrV6},
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

const GROUP: Ipv6Addr = Ipv6Addr::new(0xff12, 0, 0, 0, 0, 0, 0xe0a1, 0xde89);
const MAGIC: [u8; 3] = *b"EXO";

pub struct Discovery {
    sock: Arc<UdpSocket>,
    ifaces: Arc<Mutex<Vec<SocketAddr>>>,
    last_nonce: Mutex<[u8; 8]>,
    /// the port of the service we are doing discovery for - transmitted to peers
    listen_port: u16,
    zid: ZenohId,
    tick: Interval,
    _sync: Mutex<WatchHandle>,
}

impl Discovery {
    pub async fn new(zid: ZenohId, listen_port: u16) -> io::Result<Self> {
        let discovery_port = 52413;
        let sock = Arc::new(UdpSocket::bind(format!("[::]:{discovery_port}")).await?);
        //sock.set_multicast_loop_v6(false)?;
        let ifaces: Arc<Mutex<Vec<SocketAddr>>> = Default::default();
        let _sync = Mutex::new(
            netwatcher::watch_interfaces_with_callback({
                let sock = sock.clone();
                let ifaces = ifaces.clone();
                move |update| {
                    for (iface_idx, iface) in update.interfaces.iter() {
                        if iface
                            .ipv6_ips()
                            .all(|addr| addr.is_loopback() || addr.is_unspecified())
                        {
                            continue;
                        }
                        if let Err(e) = sock.join_multicast_v6(&GROUP, *iface_idx).inspect(|_| {
                            ifaces.lock().push(SocketAddr::V6(SocketAddrV6::new(
                                GROUP, 52413, 0, *iface_idx,
                            )))
                        }) {
                            if let Some(iface) = update.interfaces.get(&iface_idx) {
                                warn!(
                                    "failed to join multicast v6 for interface {}: {e}",
                                    iface.name
                                )
                            }
                        }
                    }
                    for iface_idx in update.diff.removed {
                        ifaces.lock().retain(|addr| {
                            if let SocketAddr::V6(v6) = addr {
                                v6.scope_id() != iface_idx
                            } else {
                                true
                            }
                        });
                        if let Err(e) = sock.leave_multicast_v6(&GROUP, iface_idx) {
                            if let Some(iface) = update.interfaces.get(&iface_idx) {
                                warn!(
                                    "failed to leave multicast v6 for interface {}: {e}",
                                    iface.name
                                )
                            }
                        }
                    }
                }
            })
            // todo: better error handling here
            .expect("failed to bind discovery watcher"),
        );
        Ok(Self {
            sock,
            ifaces,
            last_nonce: Mutex::new(rand::random()),
            listen_port,
            zid,
            tick: interval(Duration::from_secs(1)),
            _sync,
        })
    }

    pub async fn next(&mut self) -> io::Result<Discovered> {
        let mut buf = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        loop {
            tokio::select! {
                _ = self.tick.tick() => {
                    self.announce().await?;
                }
                res = self.sock.recv_from(&mut buf) => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    if let Some(discovered) = self.respond(bytes_read, addr, &buf).await? {
                        return Ok(discovered)
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

                // reply
                trace!("replying to Hello({:?})", hello.nonce);
                let mut reply_buf = [0u8; WhatsUp::buf_size()];
                WhatsUp {
                    nonce: hello.nonce,
                    zid: self.zid.to_le_bytes(),
                    port_le: self.listen_port.to_le_bytes(),
                }
                .write_into(&mut reply_buf);

                for i in 1..6 {
                    if self
                        .sock
                        .send_to(&reply_buf, addr)
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
                let SocketAddr::V6(v6) = addr else {
                    trace!("dropped: v4 addr used");
                    return Ok(None);
                };
                let Ok(zid) = ZenohId::try_from(&whats_up.zid[..]) else {
                    trace!("dropped: zenoh conversion failed");
                    return Ok(None);
                };
                if zid == self.zid {
                    trace!("dropped: self zenoh id");
                    return Ok(None);
                }
                // discovery success!
                // the incoming port is our listen port;
                // overwrite it with the whats_up port corresponding to the remote zenoh service
                let addr = {
                    let mut x = v6;
                    x.set_port(u16::from_le_bytes(whats_up.port_le));
                    x
                };
                Ok(Some(Discovered { addr, zid }))
            }
        }
    }

    async fn announce(&self) -> io::Result<()> {
        let mut buf = [0u8; Hello::buf_size()];
        let nonce = rand::random();
        *self.last_nonce.lock() = nonce;
        Hello { nonce }.write_into(&mut buf);

        let addrs = self.ifaces.lock().clone();
        debug!("announcing Hello({nonce:?}) to {addrs:?}");
        // rev so .remove() doesn't break things
        for (i, addr) in addrs.into_iter().enumerate().rev() {
            match self.sock.send_to(&buf, addr).await {
                Ok(bytes) => trace!("sent {bytes} to {addr}"),
                Err(e) if e.kind() == ErrorKind::HostUnreachable => {
                    debug!("disabling discovery address {addr}: {e}");
                    _ = self.ifaces.lock().swap_remove(i);
                }
                Err(e) => debug!("failed to reach {addr}: {e}"),
            }
        }
        Ok(())
    }
}

pub trait Message: Pod {
    const KIND: Kind;
    fn header() -> Header {
        Header {
            magic: MAGIC,
            kind: Self::KIND as u8,
        }
    }
    fn write_into(&self, buf: &mut [u8]) {
        let total = size_of::<Header>() + size_of::<Self>();
        assert!(total <= buf.len());
        buf[0..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&Self::header()));
        buf[size_of::<Header>()..total].copy_from_slice(bytemuck::bytes_of(self));
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
// packet & version
pub enum Kind {
    Hello = 0,
    WhatsUp = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct Discovered {
    pub zid: ZenohId,
    pub addr: SocketAddrV6,
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
}
impl Hello {
    const fn buf_size() -> usize {
        size_of::<Header>() + size_of::<Self>()
    }
}
impl Message for Hello {
    const KIND: Kind = Kind::Hello;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhatsUp {
    pub nonce: [u8; 8],
    pub zid: [u8; 16],
    pub port_le: [u8; 2],
}
impl WhatsUp {
    const fn buf_size() -> usize {
        size_of::<Header>() + size_of::<Self>()
    }
}
impl Message for WhatsUp {
    const KIND: Kind = Kind::WhatsUp;
}
