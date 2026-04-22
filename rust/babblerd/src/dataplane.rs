//! Dedicated dataplane thread for packet forwarding.
//!
//! The control plane remains on Tokio.
//! The hot path lives on a dedicated OS thread with:
//!
//! - immutable [`crate::fib::FibSnapshot`] swaps over `crossbeam-channel`,
//! - `mio` for readiness polling,
//! - `socket2` for UDP socket creation and interface binding,
//! - `slab` for token-indexed UDP socket storage.
//!
//! Unlike the first scaffold, interface identity here is stable across
//! snapshots:
//!
//! - the FIB keys routes by interface name,
//! - the dataplane owns a long-lived socket registry keyed by that name,
//! - and snapshot application reconciles the registry rather than trusting
//!   snapshot-local slot numbers.

use std::io::{self, ErrorKind};
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use arrayvec::ArrayVec;
use color_eyre::eyre::{Result, WrapErr, eyre};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use hashbrown::{HashMap, HashSet};
use mio::net::UdpSocket as MioUdpSocket;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use nix::errno::Errno;
use nix::net::if_::if_nametoindex;
use slab::Slab;
use socket2::{Domain, Protocol, Socket, Type};
use tun_rs::SyncDevice;

use crate::config::PHYSICAL_LINK_MTU;
use crate::fib::{FibEntry, FibSnapshot};

const TOKEN_TUN: Token = Token(0);
const MAX_SNAPSHOT_BACKLOG: usize = 8;
const POLL_INTERVAL: Duration = Duration::from_millis(250);

pub struct DataplaneConfig {
    pub tun_device: SyncDevice,
    pub udp_port: u16,
    pub initial_fib: Arc<FibSnapshot>,
}

pub struct Dataplane {
    publisher: DataplanePublisher,
    stop_send: Sender<()>,
    thread: Option<JoinHandle<Result<()>>>,
}

#[derive(Clone)]
pub struct DataplanePublisher {
    fib_updates: Sender<Arc<FibSnapshot>>,
}

pub enum PublishSnapshotError {
    Full(Arc<FibSnapshot>),
    Stopped,
}

struct DataplaneWorker {
    tun_device: SyncDevice,
    udp_port: u16,
    fib: Arc<FibSnapshot>,
    fib_updates: Receiver<Arc<FibSnapshot>>,
    stop_recv: Receiver<()>,
    poll: Poll,
    events: Events,
    sockets: Slab<InterfaceSocket>,
    ifname_to_slab: HashMap<Box<str>, usize>,
}

struct InterfaceSocket {
    ifname: Box<str>,
    ifindex: u32,
    socket: MioUdpSocket,
}

impl Dataplane {
    pub fn spawn(config: DataplaneConfig) -> Result<Self> {
        let (fib_updates_send, fib_updates_recv) = bounded(1);
        let (stop_send, stop_recv) = bounded(1);

        let thread = thread::Builder::new()
            .name("babblerd-dataplane".into())
            .spawn(move || {
                let mut worker = DataplaneWorker::new(config, fib_updates_recv, stop_recv)?;
                worker.run()
            })
            .wrap_err("spawning dataplane thread")?;

        Ok(Self {
            publisher: DataplanePublisher {
                fib_updates: fib_updates_send,
            },
            stop_send,
            thread: Some(thread),
        })
    }

    pub fn publisher(&self) -> DataplanePublisher {
        self.publisher.clone()
    }

    pub fn stop(mut self) -> Result<()> {
        let _ = self.stop_send.send(());
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        thread
            .join()
            .map_err(|_| eyre!("dataplane thread panicked"))?
    }
}

impl DataplanePublisher {
    pub fn try_publish(
        &self,
        snapshot: Arc<FibSnapshot>,
    ) -> std::result::Result<(), PublishSnapshotError> {
        match self.fib_updates.try_send(snapshot) {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Full(snapshot)) => {
                Err(PublishSnapshotError::Full(snapshot))
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(PublishSnapshotError::Stopped)
            }
        }
    }
}

impl DataplaneWorker {
    fn new(
        config: DataplaneConfig,
        fib_updates: Receiver<Arc<FibSnapshot>>,
        stop_recv: Receiver<()>,
    ) -> Result<Self> {
        let poll = Poll::new().wrap_err("creating dataplane poller")?;
        let events = Events::with_capacity(128);

        let tun_rawfd = config.tun_device.as_raw_fd();
        let mut tun_source = SourceFd(&tun_rawfd);
        poll.registry()
            .register(&mut tun_source, TOKEN_TUN, Interest::READABLE)
            .wrap_err("registering TUN fd with dataplane poller")?;

        let mut worker = Self {
            tun_device: config.tun_device,
            udp_port: config.udp_port,
            fib: config.initial_fib,
            fib_updates,
            stop_recv,
            poll,
            events,
            sockets: Slab::new(),
            ifname_to_slab: HashMap::new(),
        };
        worker.reconcile_sockets()?;
        Ok(worker)
    }

    fn run(&mut self) -> Result<()> {
        loop {
            if self.stop_requested()? {
                return Ok(());
            }

            self.drain_snapshot_updates()?;

            self.poll
                .poll(&mut self.events, Some(POLL_INTERVAL))
                .wrap_err("polling dataplane fds")?;

            let ready: Vec<Token> = self.events.iter().map(|event| event.token()).collect();
            for token in ready {
                match token {
                    TOKEN_TUN => self.handle_tun_ready()?,
                    Token(n) => {
                        let slab_key = n
                            .checked_sub(1)
                            .ok_or_else(|| eyre!("invalid UDP socket token"))?;
                        self.handle_udp_ready(slab_key)?;
                    }
                }
            }
        }
    }

    fn stop_requested(&self) -> Result<bool> {
        match self.stop_recv.try_recv() {
            Ok(()) => Ok(true),
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => Err(eyre!("dataplane stop channel closed")),
        }
    }

    fn drain_snapshot_updates(&mut self) -> Result<()> {
        let mut drained = ArrayVec::<Arc<FibSnapshot>, MAX_SNAPSHOT_BACKLOG>::new();
        loop {
            match self.fib_updates.try_recv() {
                Ok(snapshot) => {
                    if drained.is_full() {
                        drained.remove(0);
                    }
                    drained.push(snapshot);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        if let Some(snapshot) = drained.pop() {
            self.apply_snapshot(snapshot)?;
        }
        Ok(())
    }

    fn handle_tun_ready(&mut self) -> Result<()> {
        let mut buf = [0u8; PHYSICAL_LINK_MTU as usize];
        let packet_len = match nix::unistd::read(&self.tun_device, &mut buf) {
            Ok(len) => len,
            Err(Errno::EAGAIN) => return Ok(()),
            Err(err) => return Err(err).wrap_err("reading inner packet from TUN"),
        };
        if packet_len == 0 {
            return Err(eyre!("TUN device closed"));
        }

        let packet = &buf[..packet_len];
        let Some(dst) = ipv6_destination(packet) else {
            tracing::debug!("dropping invalid inner packet from TUN");
            return Ok(());
        };
        if self.fib.is_local(dst) {
            tracing::debug!(destination = %dst, "dropping self-directed packet from TUN");
            return Ok(());
        }

        let Some(route) = self.fib.lookup(dst).cloned() else {
            tracing::debug!(destination = %dst, "no dataplane route for packet from TUN");
            return Ok(());
        };
        self.send_via_route(packet, &route)
    }

    fn handle_udp_ready(&mut self, slab_key: usize) -> Result<()> {
        let Some(socket) = self.sockets.get_mut(slab_key) else {
            return Err(eyre!("unknown UDP socket token"));
        };

        let mut buf = [0u8; PHYSICAL_LINK_MTU as usize];
        let (packet_len, peer) = match socket.socket.recv_from(&mut buf) {
            Ok(res) => res,
            Err(err) if err.kind() == ErrorKind::WouldBlock => return Ok(()),
            Err(err) => {
                return Err(err).wrap_err_with(|| format!("receiving on {}", socket.ifname));
            }
        };

        let mut packet = buf[..packet_len].to_vec();
        let Some(dst) = ipv6_destination(&packet) else {
            tracing::debug!(peer = %peer, "dropping invalid UDP payload");
            return Ok(());
        };

        if self.fib.is_local(dst) {
            write_tun(&self.tun_device, &packet)?;
            return Ok(());
        }

        if !decrement_hop_limit(&mut packet) {
            tracing::debug!(destination = %dst, "dropping packet with exhausted hop limit");
            return Ok(());
        }

        let Some(route) = self.fib.lookup(dst).cloned() else {
            tracing::debug!(destination = %dst, "no dataplane route for forwarded packet");
            return Ok(());
        };
        self.send_via_route(&packet, &route)
    }

    fn send_via_route(&mut self, packet: &[u8], route: &FibEntry) -> Result<()> {
        let Some(&slab_key) = self.ifname_to_slab.get(route.ifname.as_ref()) else {
            return Err(eyre!("missing UDP socket for interface {}", route.ifname));
        };
        let Some(socket) = self.sockets.get_mut(slab_key) else {
            return Err(eyre!("stale UDP socket slot {}", slab_key));
        };

        let peer = SocketAddr::V6(SocketAddrV6::new(
            route.next_hop_ll,
            self.udp_port,
            0,
            socket.ifindex,
        ));
        match socket.socket.send_to(packet, peer) {
            Ok(_) => Ok(()),
            Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(()),
            Err(err) => Err(err).wrap_err_with(|| format!("sending packet via {}", socket.ifname)),
        }
    }

    fn apply_snapshot(&mut self, snapshot: Arc<FibSnapshot>) -> Result<()> {
        self.fib = snapshot;
        self.reconcile_sockets()
    }

    fn reconcile_sockets(&mut self) -> Result<()> {
        let required: HashSet<Box<str>> = self
            .fib
            .routes
            .values()
            .map(|route| route.ifname.clone())
            .collect();

        let stale: Vec<Box<str>> = self
            .ifname_to_slab
            .keys()
            .filter(|ifname| !required.contains(*ifname))
            .cloned()
            .collect();
        for ifname in stale {
            self.remove_socket(&ifname)?;
        }

        for ifname in required {
            if self.ifname_to_slab.contains_key(ifname.as_ref()) {
                continue;
            }
            self.add_socket(ifname)?;
        }

        Ok(())
    }

    fn add_socket(&mut self, ifname: Box<str>) -> Result<()> {
        let ifindex = if_nametoindex(ifname.as_ref())
            .wrap_err_with(|| format!("resolving ifindex for {}", ifname))?;
        let socket = open_udp_socket(self.udp_port, ifname.as_ref(), ifindex)
            .wrap_err_with(|| format!("opening UDP socket for {}", ifname))?;

        let slab_key = self.sockets.insert(InterfaceSocket {
            ifname: ifname.clone(),
            ifindex,
            socket,
        });
        let token = Token(slab_key + 1);
        let socket_ref = self
            .sockets
            .get_mut(slab_key)
            .expect("slab entry must exist immediately after insert");
        self.poll
            .registry()
            .register(&mut socket_ref.socket, token, Interest::READABLE)
            .wrap_err_with(|| format!("registering UDP socket for {}", ifname))?;
        self.ifname_to_slab.insert(ifname, slab_key);
        Ok(())
    }

    fn remove_socket(&mut self, ifname: &str) -> Result<()> {
        let Some(slab_key) = self.ifname_to_slab.remove(ifname) else {
            return Ok(());
        };
        let Some(mut socket) = self.sockets.try_remove(slab_key) else {
            return Ok(());
        };
        self.poll
            .registry()
            .deregister(&mut socket.socket)
            .wrap_err_with(|| format!("deregistering UDP socket for {}", ifname))?;
        Ok(())
    }
}

fn open_udp_socket(port: u16, ifname: &str, ifindex: u32) -> io::Result<MioUdpSocket> {
    let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    ))]
    socket.set_reuse_port(true)?;
    socket.set_only_v6(true)?;
    socket.set_nonblocking(true)?;

    let Some(ifindex) = NonZeroU32::new(ifindex) else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("invalid ifindex for {ifname}"),
        ));
    };

    #[cfg(any(target_os = "linux", target_os = "android"))]
    socket.bind_device(Some(ifname.as_bytes()))?;
    socket.bind_device_by_index_v6(Some(ifindex))?;
    socket.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, port, 0, 0).into())?;

    let udp: UdpSocket = socket.into();
    Ok(MioUdpSocket::from_std(udp))
}

fn write_tun(tun_device: &SyncDevice, packet: &[u8]) -> Result<()> {
    match nix::unistd::write(tun_device, packet) {
        Ok(_) => Ok(()),
        Err(Errno::EAGAIN) => Ok(()),
        Err(err) => Err(err).wrap_err("writing inner packet to TUN"),
    }
}

fn ipv6_destination(packet: &[u8]) -> Option<Ipv6Addr> {
    if packet.len() < 40 {
        return None;
    }
    if packet[0] >> 4 != 6 {
        return None;
    }
    let dst = <[u8; 16]>::try_from(&packet[24..40]).ok()?;
    Some(Ipv6Addr::from(dst))
}

fn decrement_hop_limit(packet: &mut [u8]) -> bool {
    if packet.len() < 40 || packet[0] >> 4 != 6 {
        return false;
    }
    let hop_limit = &mut packet[7];
    if *hop_limit <= 1 {
        return false;
    }
    *hop_limit -= 1;
    true
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;

    use super::{decrement_hop_limit, ipv6_destination};

    fn sample_ipv6_packet(dst: Ipv6Addr, hop_limit: u8) -> Vec<u8> {
        let mut packet = vec![0u8; 40];
        packet[0] = 0x60;
        packet[6] = 17;
        packet[7] = hop_limit;
        packet[24..40].copy_from_slice(&dst.octets());
        packet
    }

    #[test]
    fn parses_destination_address() {
        let packet = sample_ipv6_packet("fde0::1234".parse().unwrap(), 32);
        assert_eq!(
            ipv6_destination(&packet),
            Some("fde0::1234".parse::<Ipv6Addr>().unwrap())
        );
    }

    #[test]
    fn decrements_hop_limit_until_drop() {
        let mut packet = sample_ipv6_packet("fde0::1234".parse().unwrap(), 2);
        assert!(decrement_hop_limit(&mut packet));
        assert_eq!(packet[7], 1);
        assert!(!decrement_hop_limit(&mut packet));
    }
}
