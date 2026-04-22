//! Dedicated dataplane thread for packet forwarding.
//!
//! The control plane remains on Tokio.
//! The hot path lives on a dedicated OS thread with:
//!
//! - immutable [`crate::fib::FibSnapshot`] swaps over `crossbeam-channel`,
//! - `mio` for readiness polling,
//! - `socket2` for UDP socket creation,
//! - `slab` for token-indexed interface socket storage.
//!
//! This module is intentionally not wired into the daemon yet. The goal of
//! this step is to land the hot-path structure and data ownership model before
//! threading it through the resident daemon shell.

use std::io::{self, ErrorKind};
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use arrayvec::ArrayVec;
use color_eyre::eyre::{Result, WrapErr, eyre};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use hashbrown::HashMap;
use mio::net::UdpSocket as MioUdpSocket;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use nix::errno::Errno;
use slab::Slab;
use socket2::{Domain, Protocol, Socket, Type};

use crate::config::PHYSICAL_LINK_MTU;
use crate::fib::{FibEntry, FibSnapshot};

const TOKEN_TUN: Token = Token(0);
const MAX_SNAPSHOT_BACKLOG: usize = 8;
const POLL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
pub struct InterfaceSocketConfig {
    pub if_slot: u16,
    pub ifname: Box<str>,
    pub ifindex: u32,
}

#[derive(Debug)]
pub struct DataplaneConfig {
    pub tun_fd: OwnedFd,
    pub udp_port: u16,
    pub initial_fib: Arc<FibSnapshot>,
    pub interfaces: Vec<InterfaceSocketConfig>,
}

pub struct Dataplane {
    fib_updates: Sender<Arc<FibSnapshot>>,
    stop_send: Sender<()>,
    thread: Option<JoinHandle<Result<()>>>,
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
            fib_updates: fib_updates_send,
            stop_send,
            thread: Some(thread),
        })
    }

    pub fn snapshot_sender(&self) -> Sender<Arc<FibSnapshot>> {
        self.fib_updates.clone()
    }

    pub fn replace_snapshot(&self, snapshot: Arc<FibSnapshot>) -> Result<()> {
        match self.fib_updates.try_send(snapshot) {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Full(_)) => Err(eyre!(
                "failed to publish dataplane snapshot: dataplane is still processing the previous update"
            )),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(eyre!("dataplane thread stopped"))
            }
        }
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

struct DataplaneWorker {
    tun_fd: OwnedFd,
    udp_port: u16,
    fib: Arc<FibSnapshot>,
    fib_updates: Receiver<Arc<FibSnapshot>>,
    stop_recv: Receiver<()>,
    poll: Poll,
    events: Events,
    sockets: Slab<InterfaceSocket>,
    if_slot_to_slab: HashMap<u16, usize>,
}

struct InterfaceSocket {
    if_slot: u16,
    ifname: Box<str>,
    ifindex: u32,
    socket: MioUdpSocket,
}

impl DataplaneWorker {
    fn new(
        config: DataplaneConfig,
        fib_updates: Receiver<Arc<FibSnapshot>>,
        stop_recv: Receiver<()>,
    ) -> Result<Self> {
        let poll = Poll::new().wrap_err("creating dataplane poller")?;
        let events = Events::with_capacity(128);

        let tun_rawfd = config.tun_fd.as_raw_fd();
        let mut tun_source = SourceFd(&tun_rawfd);
        poll.registry()
            .register(&mut tun_source, TOKEN_TUN, Interest::READABLE)
            .wrap_err("registering TUN fd with dataplane poller")?;

        let mut sockets = Slab::with_capacity(config.interfaces.len());
        let mut if_slot_to_slab = HashMap::with_capacity(config.interfaces.len());
        for iface in config.interfaces {
            let socket = open_udp_socket(config.udp_port)
                .wrap_err_with(|| format!("opening UDP socket for {}", iface.ifname))?;
            let slab_key = sockets.insert(InterfaceSocket {
                if_slot: iface.if_slot,
                ifname: iface.ifname,
                ifindex: iface.ifindex,
                socket,
            });
            let token = Token(slab_key + 1);
            let iface_socket = sockets
                .get_mut(slab_key)
                .expect("slab entry should exist immediately after insert");
            poll.registry()
                .register(&mut iface_socket.socket, token, Interest::READABLE)
                .wrap_err_with(|| format!("registering UDP socket for {}", iface_socket.ifname))?;
            if_slot_to_slab.insert(iface_socket.if_slot, slab_key);
        }

        Ok(Self {
            tun_fd: config.tun_fd,
            udp_port: config.udp_port,
            fib: config.initial_fib,
            fib_updates,
            stop_recv,
            poll,
            events,
            sockets,
            if_slot_to_slab,
        })
    }

    fn run(&mut self) -> Result<()> {
        loop {
            if self.stop_requested()? {
                return Ok(());
            }

            self.drain_snapshot_updates();

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

    fn drain_snapshot_updates(&mut self) {
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
            self.fib = snapshot;
        }
    }

    fn handle_tun_ready(&mut self) -> Result<()> {
        let mut buf = [0u8; PHYSICAL_LINK_MTU as usize];
        let packet_len = match nix::unistd::read(&self.tun_fd, &mut buf) {
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
            tracing::debug!(destination=%dst, "dropping self-directed packet from TUN");
            return Ok(());
        }

        let Some(route) = self.fib.lookup(dst).cloned() else {
            tracing::debug!(destination=%dst, "no dataplane route for packet from TUN");
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
            tracing::debug!(peer=%peer, "dropping invalid UDP payload");
            return Ok(());
        };

        if self.fib.is_local(dst) {
            write_tun(&self.tun_fd, &packet)?;
            return Ok(());
        }

        if !decrement_hop_limit(&mut packet) {
            tracing::debug!(destination=%dst, "dropping packet with exhausted hop limit");
            return Ok(());
        }

        let Some(route) = self.fib.lookup(dst).cloned() else {
            tracing::debug!(destination=%dst, "no dataplane route for forwarded packet");
            return Ok(());
        };
        self.send_via_route(&packet, &route)
    }

    fn send_via_route(&mut self, packet: &[u8], route: &FibEntry) -> Result<()> {
        let Some(&slab_key) = self.if_slot_to_slab.get(&route.if_slot) else {
            return Err(eyre!(
                "missing UDP socket for interface slot {}",
                route.if_slot
            ));
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
            Err(err) => Err(err).wrap_err_with(|| {
                format!(
                    "sending packet via {} (slot {})",
                    socket.ifname, socket.if_slot
                )
            }),
        }
    }
}

fn open_udp_socket(port: u16) -> io::Result<MioUdpSocket> {
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
    socket.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, port, 0, 0).into())?;

    let udp: UdpSocket = socket.into();
    Ok(MioUdpSocket::from_std(udp))
}

fn write_tun(tun_fd: &OwnedFd, packet: &[u8]) -> Result<()> {
    match nix::unistd::write(tun_fd, packet) {
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
