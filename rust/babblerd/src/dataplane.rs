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

use std::hash::BuildHasher;
use std::io::{self, ErrorKind};
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use arrayvec::ArrayVec;
use color_eyre::eyre::{eyre, Result, WrapErr};
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum SocketAction {
    Add { ifname: Box<str>, ifindex: u32 },
    Refresh { ifname: Box<str>, ifindex: u32 },
    Remove { ifname: Box<str> },
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
        let current_ifindices = self.current_ifindices();
        let actions = plan_socket_actions(
            &current_ifindices,
            &self.fib.admitted_interfaces,
            |ifname| {
                if_nametoindex(ifname).wrap_err_with(|| format!("resolving ifindex for {}", ifname))
            },
        )?;

        for action in actions {
            match action {
                SocketAction::Add { ifname, ifindex } => self.add_socket(ifname, ifindex)?,
                SocketAction::Refresh { ifname, ifindex } => {
                    self.remove_socket(&ifname)?;
                    self.add_socket(ifname, ifindex)?;
                }
                SocketAction::Remove { ifname } => self.remove_socket(&ifname)?,
            }
        }

        Ok(())
    }

    fn current_ifindices(&self) -> HashMap<Box<str>, u32> {
        self.ifname_to_slab
            .iter()
            .filter_map(|(ifname, slab_key)| {
                self.sockets
                    .get(*slab_key)
                    .map(|socket| (ifname.clone(), socket.ifindex))
            })
            .collect()
    }

    fn add_socket(&mut self, ifname: Box<str>, ifindex: u32) -> Result<()> {
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

fn plan_socket_actions<F, S1, S2>(
    current_ifindices: &HashMap<Box<str>, u32, S1>,
    admitted_interfaces: &HashSet<Box<str>, S2>,
    mut resolve_ifindex: F,
) -> Result<Vec<SocketAction>>
where
    F: FnMut(&str) -> Result<u32>,
    S1: BuildHasher,
    S2: BuildHasher,
{
    let mut actions = Vec::new();

    let mut stale: Vec<Box<str>> = current_ifindices
        .keys()
        .filter(|ifname| !admitted_interfaces.contains(*ifname))
        .cloned()
        .collect();
    stale.sort();
    for ifname in stale {
        actions.push(SocketAction::Remove { ifname });
    }

    let mut required: Vec<Box<str>> = admitted_interfaces.iter().cloned().collect();
    required.sort();
    for ifname in required {
        let ifindex = resolve_ifindex(ifname.as_ref())?;
        match current_ifindices.get(ifname.as_ref()).copied() {
            None => actions.push(SocketAction::Add { ifname, ifindex }),
            Some(existing_ifindex) if existing_ifindex != ifindex => {
                actions.push(SocketAction::Refresh { ifname, ifindex });
            }
            Some(_) => {}
        }
    }

    Ok(actions)
}

fn open_udp_socket(port: u16, ifname: &str, ifindex: u32) -> io::Result<MioUdpSocket> {
    let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_only_v6(true)?;
    socket.set_nonblocking(true)?;

    let Some(ifindex) = NonZeroU32::new(ifindex) else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("invalid ifindex for {ifname}"),
        ));
    };

    #[cfg(target_os = "linux")]
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

    use ahash::RandomState;
    use hashbrown::{HashMap, HashSet};

    use super::{decrement_hop_limit, ipv6_destination, plan_socket_actions, SocketAction};

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

    #[test]
    fn socket_plan_uses_admitted_interfaces_not_routes() {
        let current_ifindices = HashMap::with_hasher(RandomState::new());
        let admitted_interfaces: HashSet<Box<str>> = ["mesh0".into()].into_iter().collect();

        let actions = plan_socket_actions(&current_ifindices, &admitted_interfaces, |ifname| {
            Ok(match ifname {
                "mesh0" => 7,
                _ => unreachable!(),
            })
        })
        .unwrap();

        assert_eq!(
            actions,
            vec![SocketAction::Add {
                ifname: "mesh0".into(),
                ifindex: 7,
            }]
        );
    }

    #[test]
    fn socket_plan_refreshes_ifindex_changes_and_removes_stale() {
        let mut current_ifindices = HashMap::with_hasher(RandomState::new());
        current_ifindices.insert("stale0".into(), 1);
        current_ifindices.insert("mesh0".into(), 99);

        let admitted_interfaces: HashSet<Box<str>> = ["mesh0".into()].into_iter().collect();
        let actions = plan_socket_actions(&current_ifindices, &admitted_interfaces, |ifname| {
            Ok(match ifname {
                "mesh0" => 7,
                _ => unreachable!(),
            })
        })
        .unwrap();

        assert_eq!(
            actions,
            vec![
                SocketAction::Remove {
                    ifname: "stale0".into(),
                },
                SocketAction::Refresh {
                    ifname: "mesh0".into(),
                    ifindex: 7,
                },
            ]
        );
    }
}
