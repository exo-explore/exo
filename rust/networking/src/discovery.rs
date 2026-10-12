use std::{
    collections::VecDeque,
    io,
    net::{Ipv6Addr, SocketAddr, SocketAddrV6},
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use log::{debug, info, trace, warn};
use netwatcher::WatchHandle;
use parking_lot::Mutex;
use tokio::{
    net::UdpSocket,
    time::{Interval, MissedTickBehavior, interval},
};
use zenoh::config::ZenohId;

const GROUP: Ipv6Addr = Ipv6Addr::new(0xff12, 0, 0, 0, 0, 0, 0xe0a1, 0xde89);
const MAGIC: [u8; 3] = *b"EXO";
/// Replies to any of our last few Hellos are accepted. A reply can arrive after the next
/// Hello has gone out (the peer, or this loop, was busy), and insisting on the latest nonce
/// meant a slow peer was never discovered.
const RECENT_NONCES: usize = 8;
/// How often to repeat the warning while no Hello can be sent at all.
const BLOCKED_WARNING_INTERVAL: Duration = Duration::from_secs(60);
/// A pause this long (the process was suspended, the machine slept, or the runtime was
/// starved) may have let peers expire their sessions with us while we still hold ours.
const STALL_THRESHOLD: Duration = Duration::from_secs(5);
/// After such a pause we neither answer nor dial peers for this long (zenoh's lease is 10s),
/// so the stale sessions are closed before new links are made. zenoh adds a new link to a
/// peer's existing session instead of starting a new one, and when the peer had already
/// dropped that session it never receives our declarations again: a one-way split.
const STALL_HOLD_OFF: Duration = Duration::from_secs(12);

pub struct Discovery {
    sock: Arc<UdpSocket>,
    ifaces: Arc<Mutex<Vec<SocketAddrV6>>>,
    namespace: [u8; 8],
    recent_nonces: Mutex<RecentNonces>,
    /// the port of the service we are doing discovery for - transmitted to peers
    listen_port: u16,
    zid: ZenohId,
    tick: Interval,
    /// When we last warned that no Hello could be sent; `None` while sending works.
    blocked_since_warning: Mutex<Option<Instant>>,
    stall_guard: StallGuard,
    _sync: Mutex<WatchHandle>,
}

#[derive(Debug, Clone, Copy)]
pub struct Discovered {
    pub zid: ZenohId,
    pub addr: SocketAddrV6,
}

impl Discovery {
    pub async fn new(
        zid: ZenohId,
        namespace: [u8; 8],
        listen_port: u16,
        discovery_port: u16,
    ) -> io::Result<Self> {
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
        let sock = Arc::new(UdpSocket::from_std(sock.into())?);
        let ifaces: Arc<Mutex<Vec<SocketAddrV6>>> = Default::default();
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
            namespace,
            ifaces,
            recent_nonces: Mutex::new(RecentNonces::default()),
            listen_port,
            zid,
            tick: {
                // After a stall, announce once rather than in a burst of missed ticks
                let mut tick = interval(Duration::from_secs(1));
                tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
                tick
            },
            blocked_since_warning: Mutex::new(None),
            stall_guard: StallGuard::default(),
            _sync,
        })
    }

    pub async fn next(&mut self) -> io::Result<Discovered> {
        let mut buf = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        loop {
            tokio::select! {
                // The tick first: after a stall it tells us so before any queued message is handled
                biased;
                scheduled = self.tick.tick() => {
                    let late = scheduled.elapsed();
                    if self.stall_guard.on_tick(late, Instant::now()) {
                        warn!(
                            "this node was unresponsive for {late:.0?}; ignoring peers for \
                             {STALL_HOLD_OFF:?} so stale sessions close before reconnecting"
                        );
                    }
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
        &mut self,
        bytes_read: usize,
        addr: SocketAddr,
        buf: &[u8],
    ) -> io::Result<Option<Discovered>> {
        if self.stall_guard.holding_off(Instant::now()) {
            trace!("dropped: holding off after a stall");
            return Ok(None);
        }
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
                if self.recent_nonces.lock().contains(hello.nonce) {
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

                // One attempt only: retrying here would stall this loop (and with it our own
                // announcements and every other reply), and the peer says Hello again every
                // second anyway.
                match self.sock.send_to(&reply, addr).await {
                    Ok(sent) => trace!("sent {sent} bytes to {addr}"),
                    Err(e) => debug!("send to {addr} failed: {e}"),
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
                if !self.recent_nonces.lock().contains(whats_up.nonce) {
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
        let nonce = rand::random();
        self.recent_nonces.lock().push(nonce);
        let buf = Hello {
            nonce,
            namespace: self.namespace,
        }
        .alloc();

        let addrs = self.ifaces.lock().clone();
        debug!("announcing Hello({nonce:?}) to {addrs:?}");
        match send_to_all(&self.sock, &buf, &addrs).await {
            Ok(()) => {
                if self.blocked_since_warning.lock().take().is_some() {
                    info!("peer discovery can send again");
                }
            }
            Err(e) => {
                let warning_due = {
                    let mut last_warning = self.blocked_since_warning.lock();
                    let due =
                        last_warning.is_none_or(|at| at.elapsed() >= BLOCKED_WARNING_INTERVAL);
                    if due {
                        *last_warning = Some(Instant::now());
                    }
                    due
                };
                if warning_due {
                    warn!(
                        "peer discovery could not send on any network interface ({e}), so other \
                         nodes cannot find this one. On macOS this usually means Local Network \
                         access is blocked for this process: allow it in System Settings > \
                         Privacy & Security > Local Network. Background services started \
                         outside a login session can be denied without a prompt. Retrying \
                         every second."
                    );
                }
            }
        }
        Ok(())
    }
}

/// Ignores peers for a while after this process stalled (see `STALL_HOLD_OFF`).
#[derive(Debug, Default)]
struct StallGuard {
    held_off_until: Option<Instant>,
}

impl StallGuard {
    /// Called on every tick with how late it fired. Returns whether that was a stall.
    fn on_tick(&mut self, late: Duration, now: Instant) -> bool {
        if late < STALL_THRESHOLD {
            return false;
        }
        self.held_off_until = Some(now + STALL_HOLD_OFF);
        true
    }

    fn holding_off(&mut self, now: Instant) -> bool {
        match self.held_off_until {
            Some(until) if now < until => true,
            Some(_) => {
                info!("resuming peer discovery");
                self.held_off_until = None;
                false
            }
            None => false,
        }
    }
}

/// Send `buf` to every address, returning an error only if nothing could be sent.
///
/// Failed addresses are not dropped: macOS returns EHOSTUNREACH while Local Network access
/// is denied or its prompt is still pending, and an interface may simply not have a route
/// yet. Retrying on the next tick lets discovery recover on its own once sending works.
async fn send_to_all(sock: &UdpSocket, buf: &[u8], addrs: &[SocketAddrV6]) -> io::Result<()> {
    let mut last_err = None;
    let mut sent_any = false;
    for addr in addrs {
        match sock.send_to(buf, addr).await {
            Ok(bytes) => {
                sent_any = true;
                trace!("sent {bytes} to {addr}");
            }
            Err(e) => {
                debug!("failed to reach {addr}: {e}");
                last_err = Some(e);
            }
        }
    }
    match last_err {
        Some(e) if !sent_any => Err(e),
        _ => Ok(()),
    }
}

/// The nonces of our last few Hellos.
#[derive(Default)]
struct RecentNonces(VecDeque<[u8; 8]>);

impl RecentNonces {
    fn push(&mut self, nonce: [u8; 8]) {
        if self.0.len() == RECENT_NONCES {
            self.0.pop_front();
        }
        self.0.push_back(nonce);
    }

    fn contains(&self, nonce: [u8; 8]) -> bool {
        self.0.contains(&nonce)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_tick_holds_off_peers_for_a_while() {
        let mut guard = StallGuard::default();
        let start = Instant::now();
        assert!(!guard.on_tick(Duration::from_millis(1500), start));
        assert!(!guard.holding_off(start));

        assert!(guard.on_tick(Duration::from_secs(30), start));
        assert!(guard.holding_off(start));
        assert!(guard.holding_off(start + STALL_HOLD_OFF.saturating_sub(Duration::from_millis(1))));
        assert!(!guard.holding_off(start + STALL_HOLD_OFF));
        assert!(!guard.holding_off(start + STALL_HOLD_OFF * 2));
    }

    #[tokio::test(start_paused = true)]
    async fn a_stall_shows_up_as_a_late_tick() {
        let mut tick = interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        tick.tick().await;
        // Nothing polls the interval for 30s, as when the process is suspended
        tokio::time::advance(Duration::from_secs(30)).await;
        let late = tick.tick().await.elapsed();
        assert!(late >= STALL_THRESHOLD, "{late:?}");
        // The next tick is on time again
        let next_late = tick.tick().await.elapsed();
        assert!(next_late < STALL_THRESHOLD, "{next_late:?}");
    }

    #[test]
    fn remembers_only_the_most_recent_nonces() {
        let mut recent = RecentNonces::default();
        let nonces: Vec<[u8; 8]> = (0..=RECENT_NONCES as u8).map(|i| [i; 8]).collect();
        for nonce in &nonces {
            recent.push(*nonce);
        }
        assert!(!recent.contains(nonces[0]));
        assert!(nonces[1..].iter().all(|nonce| recent.contains(*nonce)));
    }

    /// An address no packet can be sent to: the discovery group on an interface index that
    /// doesn't exist.
    fn unsendable() -> SocketAddrV6 {
        SocketAddrV6::new(GROUP, 9, 0, u32::MAX)
    }

    /// A UDP socket on the IPv6 loopback, or `None` where there isn't one (some sandboxes).
    async fn local_socket() -> Option<(UdpSocket, SocketAddrV6)> {
        let sock = UdpSocket::bind("[::1]:0").await.ok()?;
        match sock.local_addr().ok()? {
            SocketAddr::V6(addr) => Some((sock, addr)),
            SocketAddr::V4(_) => None,
        }
    }

    #[tokio::test]
    async fn errors_when_nothing_can_be_sent() {
        let Some((sender, _)) = local_socket().await else {
            eprintln!("skipping: no IPv6 loopback");
            return;
        };
        assert!(
            send_to_all(&sender, b"hello", &[unsendable()])
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn one_working_address_is_enough() {
        let (Some((sender, _)), Some((receiver, receiver_addr))) =
            (local_socket().await, local_socket().await)
        else {
            eprintln!("skipping: no IPv6 loopback");
            return;
        };
        let addrs = [unsendable(), receiver_addr];

        send_to_all(&sender, b"hello", &addrs)
            .await
            .expect("send succeeds on the working address");

        let mut buf = [0u8; 5];
        let (len, _) = receiver.recv_from(&mut buf).await.expect("recv");
        assert_eq!(&buf[..len], b"hello");
    }
}
