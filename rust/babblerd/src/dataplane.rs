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
use std::io::{self, ErrorKind, IoSliceMut};
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ahash::RandomState;
use arrayvec::ArrayVec;
use color_eyre::eyre::{Result, WrapErr, eyre};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use hashbrown::{HashMap, HashSet};
use iroh_quinn_udp::{BATCH_SIZE, RecvMeta, Transmit, UdpSockRef, UdpSocketState};
use mio::net::UdpSocket as MioUdpSocket;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use nix::net::if_::if_nametoindex;
use slab::Slab;
use socket2::{Domain, Protocol, Socket, Type};
use tokio::sync::oneshot;
use tun_rs::SyncDevice;

use crate::config::PHYSICAL_LINK_MTU;
use crate::fib::{FibSnapshot, HostKey, host_key};

const TOKEN_TUN: Token = Token(0);
const MAX_POLL_EVENTS: usize = 128;
const POLL_INTERVAL: Duration = Duration::from_millis(250);
const RECONCILE_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const COUNTER_LOG_INTERVAL: Duration = Duration::from_secs(5);
const UDP_SOCKET_BUFFER_BYTES: usize = 4 * 1024 * 1024;
const TUN_DRAIN_BUDGET: usize = 64;
const UDP_DRAIN_BUDGET: usize = 64;
const PACKET_BUFFER_BYTES: usize = PHYSICAL_LINK_MTU as usize;
const MAX_GRO_SEGMENTS: usize = 64;
const UDP_RECV_BUFFER_BYTES: usize = PACKET_BUFFER_BYTES * MAX_GRO_SEGMENTS;

pub struct DataplaneConfig {
    pub tun_device: Arc<SyncDevice>,
    pub udp_port: u16,
    pub initial_fib: Arc<FibSnapshot>,
}

pub struct Dataplane {
    publisher: DataplanePublisher,
    stop_send: Sender<()>,
    exit_recv: Option<oneshot::Receiver<std::result::Result<(), String>>>,
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
    tun_device: Arc<SyncDevice>,
    udp_port: u16,
    fib: Arc<FibSnapshot>,
    fast_routes: HashMap<HostKey, FastFibEntry, RandomState>,
    fib_updates: Receiver<Arc<FibSnapshot>>,
    stop_recv: Receiver<()>,
    poll: Poll,
    events: Events,
    sockets: Slab<InterfaceSocket>,
    ifname_to_slab: HashMap<Box<str>, usize>,
    needs_reconcile_retry: bool,
    last_reconcile_attempt: Instant,
    counters: DataplaneCounters,
    last_counter_log: Instant,
    last_logged_counters: DataplaneCounters,
    tun_buf: Box<[u8; PACKET_BUFFER_BYTES]>,
    udp_batch: UdpRecvBatch,
}

struct InterfaceSocket {
    ifname: Box<str>,
    ifindex: u32,
    socket: MioUdpSocket,
    udp_state: UdpSocketState,
}

struct UdpRecvBatch {
    buffers: Vec<Box<[u8]>>,
    metas: Vec<RecvMeta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FastFibEntry {
    next_hop_ll: Ipv6Addr,
    socket_slot: usize,
    mtu: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacketWriteOutcome {
    Sent(usize),
    WouldBlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UdpRecvOutcome {
    Packets(usize),
    WouldBlock,
    Error,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct DataplaneCounters {
    tun_rx_packets: u64,
    tun_rx_bytes: u64,
    udp_rx_packets: u64,
    udp_rx_bytes: u64,
    udp_tx_packets: u64,
    udp_tx_bytes: u64,
    tun_tx_packets: u64,
    tun_tx_bytes: u64,
    tun_to_udp_packets: u64,
    udp_forwarded_packets: u64,
    local_delivered_packets: u64,
    no_route_drops: u64,
    invalid_packet_drops: u64,
    self_directed_drops: u64,
    hop_limit_drops: u64,
    udp_send_would_block_drops: u64,
    tun_send_would_block_drops: u64,
    udp_send_errors: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SocketAction {
    Add { ifname: Box<str>, ifindex: u32 },
    Refresh { ifname: Box<str>, ifindex: u32 },
    Remove { ifname: Box<str> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkippedInterface {
    ifname: Box<str>,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SocketPlan {
    actions: Vec<SocketAction>,
    skipped: Vec<SkippedInterface>,
}

impl DataplaneCounters {
    fn saturating_delta(self, previous: Self) -> Self {
        Self {
            tun_rx_packets: self.tun_rx_packets.saturating_sub(previous.tun_rx_packets),
            tun_rx_bytes: self.tun_rx_bytes.saturating_sub(previous.tun_rx_bytes),
            udp_rx_packets: self.udp_rx_packets.saturating_sub(previous.udp_rx_packets),
            udp_rx_bytes: self.udp_rx_bytes.saturating_sub(previous.udp_rx_bytes),
            udp_tx_packets: self.udp_tx_packets.saturating_sub(previous.udp_tx_packets),
            udp_tx_bytes: self.udp_tx_bytes.saturating_sub(previous.udp_tx_bytes),
            tun_tx_packets: self.tun_tx_packets.saturating_sub(previous.tun_tx_packets),
            tun_tx_bytes: self.tun_tx_bytes.saturating_sub(previous.tun_tx_bytes),
            tun_to_udp_packets: self
                .tun_to_udp_packets
                .saturating_sub(previous.tun_to_udp_packets),
            udp_forwarded_packets: self
                .udp_forwarded_packets
                .saturating_sub(previous.udp_forwarded_packets),
            local_delivered_packets: self
                .local_delivered_packets
                .saturating_sub(previous.local_delivered_packets),
            no_route_drops: self.no_route_drops.saturating_sub(previous.no_route_drops),
            invalid_packet_drops: self
                .invalid_packet_drops
                .saturating_sub(previous.invalid_packet_drops),
            self_directed_drops: self
                .self_directed_drops
                .saturating_sub(previous.self_directed_drops),
            hop_limit_drops: self
                .hop_limit_drops
                .saturating_sub(previous.hop_limit_drops),
            udp_send_would_block_drops: self
                .udp_send_would_block_drops
                .saturating_sub(previous.udp_send_would_block_drops),
            tun_send_would_block_drops: self
                .tun_send_would_block_drops
                .saturating_sub(previous.tun_send_would_block_drops),
            udp_send_errors: self
                .udp_send_errors
                .saturating_sub(previous.udp_send_errors),
        }
    }

    fn has_activity(self) -> bool {
        self.tun_rx_packets != 0
            || self.udp_rx_packets != 0
            || self.udp_tx_packets != 0
            || self.tun_tx_packets != 0
            || self.no_route_drops != 0
            || self.invalid_packet_drops != 0
            || self.self_directed_drops != 0
            || self.hop_limit_drops != 0
            || self.udp_send_would_block_drops != 0
            || self.tun_send_would_block_drops != 0
            || self.udp_send_errors != 0
    }
}

impl UdpRecvBatch {
    fn new() -> Self {
        let buffers = (0..BATCH_SIZE)
            .map(|_| vec![0u8; UDP_RECV_BUFFER_BYTES].into_boxed_slice())
            .collect();
        let metas = vec![RecvMeta::default(); BATCH_SIZE];
        Self { buffers, metas }
    }

    fn recv_from(&mut self, socket: &InterfaceSocket, max_packets: usize) -> io::Result<usize> {
        let batch_len = max_packets.min(BATCH_SIZE).min(self.buffers.len());
        if batch_len == 0 {
            return Ok(0);
        }

        // The socket is nonblocking. OS batch receive returns the packets already
        // queued for this fd; it must not wait for `batch_len` packets.
        let mut slices = ArrayVec::<IoSliceMut<'_>, BATCH_SIZE>::new();
        for buffer in self.buffers.iter_mut().take(batch_len) {
            slices.push(IoSliceMut::new(buffer.as_mut()));
        }

        socket.socket.try_io(|| {
            socket.udp_state.recv(
                UdpSockRef::from(&socket.socket),
                slices.as_mut_slice(),
                &mut self.metas[..batch_len],
            )
        })
    }
}

fn udp_meta_stride(meta: RecvMeta) -> usize {
    if meta.stride == 0 || meta.stride > meta.len {
        meta.len.max(1)
    } else {
        meta.stride
    }
}

impl Dataplane {
    pub fn spawn(config: DataplaneConfig) -> Result<Self> {
        let (fib_updates_send, fib_updates_recv) = bounded(1);
        let (stop_send, stop_recv) = bounded(1);
        let (exit_send, exit_recv) = oneshot::channel();

        let thread = thread::Builder::new()
            .name("babblerd-dataplane".into())
            .spawn(move || {
                let mut worker = DataplaneWorker::new(config, fib_updates_recv, stop_recv)?;
                let result = worker.run();
                let _ = exit_send.send(
                    result
                        .as_ref()
                        .map(|_| ())
                        .map_err(std::string::ToString::to_string),
                );
                result
            })
            .wrap_err("spawning dataplane thread")?;

        Ok(Self {
            publisher: DataplanePublisher {
                fib_updates: fib_updates_send,
            },
            stop_send,
            exit_recv: Some(exit_recv),
            thread: Some(thread),
        })
    }

    pub fn publisher(&self) -> DataplanePublisher {
        self.publisher.clone()
    }

    pub fn take_exit_receiver(
        &mut self,
    ) -> Option<oneshot::Receiver<std::result::Result<(), String>>> {
        self.exit_recv.take()
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
            fast_routes: HashMap::with_hasher(RandomState::new()),
            fib_updates,
            stop_recv,
            poll,
            events,
            sockets: Slab::new(),
            ifname_to_slab: HashMap::new(),
            needs_reconcile_retry: false,
            last_reconcile_attempt: Instant::now(),
            counters: DataplaneCounters::default(),
            last_counter_log: Instant::now(),
            last_logged_counters: DataplaneCounters::default(),
            tun_buf: Box::new([0u8; PACKET_BUFFER_BYTES]),
            udp_batch: UdpRecvBatch::new(),
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
            self.maybe_retry_reconcile();

            self.poll
                .poll(&mut self.events, Some(POLL_INTERVAL))
                .wrap_err("polling dataplane fds")?;

            let mut ready = ArrayVec::<Token, MAX_POLL_EVENTS>::new();
            for event in self.events.iter() {
                if ready.try_push(event.token()).is_err() {
                    break;
                }
            }
            for token in ready {
                match token {
                    TOKEN_TUN => {
                        tracing::trace!("dataplane poll signaled TUN readable");
                        self.drain_tun_ready()?
                    }
                    Token(n) => {
                        let slab_key = n
                            .checked_sub(1)
                            .ok_or_else(|| eyre!("invalid UDP socket token"))?;
                        self.drain_udp_ready(slab_key)?;
                    }
                }
            }

            self.maybe_log_counters();
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
        if let Some(snapshot) = self.fib_updates.try_iter().last() {
            self.apply_snapshot(snapshot)?;
        }
        Ok(())
    }

    fn drain_tun_ready(&mut self) -> Result<()> {
        for _ in 0..TUN_DRAIN_BUDGET {
            if !self.handle_one_tun_packet()? {
                break;
            }
        }
        Ok(())
    }

    fn drain_udp_ready(&mut self, slab_key: usize) -> Result<()> {
        let mut remaining_budget = UDP_DRAIN_BUDGET;
        while remaining_budget > 0 {
            let received_buffers = match self.recv_udp_batch(slab_key, remaining_budget)? {
                Some(received_buffers) => received_buffers,
                None => break,
            };
            let processed_packets = self.handle_udp_batch(received_buffers)?;
            if processed_packets == 0 {
                break;
            }
            remaining_budget = remaining_budget.saturating_sub(processed_packets);
        }
        Ok(())
    }

    fn handle_one_tun_packet(&mut self) -> Result<bool> {
        let packet_len = match self.tun_device.recv(self.tun_buf.as_mut()) {
            Ok(len) => len,
            Err(err) if err.kind() == ErrorKind::WouldBlock => return Ok(false),
            Err(err) => return Err(err).wrap_err("reading inner packet from TUN"),
        };
        if packet_len == 0 {
            return Err(eyre!("TUN device closed"));
        }
        self.counters.tun_rx_packets += 1;
        self.counters.tun_rx_bytes += packet_len as u64;

        tracing::debug!(bytes = packet_len, "dataplane read inner packet from TUN");

        let packet = &self.tun_buf[..packet_len];
        let Some(dst) = ipv6_destination(packet) else {
            self.counters.invalid_packet_drops += 1;
            tracing::debug!("dropping invalid inner packet from TUN");
            return Ok(true);
        };
        tracing::debug!(bytes = packet_len, destination = %dst, "dataplane parsed TUN packet destination");
        if self.is_local(dst) {
            self.counters.self_directed_drops += 1;
            tracing::debug!(destination = %dst, "dataplane dropping self-directed packet from TUN");
            return Ok(true);
        }

        let Some(route) = self.lookup_fast_route(dst) else {
            self.counters.no_route_drops += 1;
            tracing::debug!(destination = %dst, "dataplane has no route for packet from TUN");
            return Ok(true);
        };
        tracing::debug!(
            destination = %dst,
            socket_slot = route.socket_slot,
            next_hop = %route.next_hop_ll,
            mtu = route.mtu,
            "dataplane resolved route for TUN packet"
        );
        let send_result = send_via_route(&self.sockets, self.udp_port, packet, route);
        if self.handle_udp_send_result(send_result, route, "sending packet from TUN")? {
            self.counters.tun_to_udp_packets += 1;
        }
        Ok(true)
    }

    fn recv_udp_batch(&mut self, slab_key: usize, max_packets: usize) -> Result<Option<usize>> {
        let Some(socket) = self.sockets.get(slab_key) else {
            return Err(eyre!("unknown UDP socket token"));
        };

        let received_buffers = match recv_udp_packets(socket, &mut self.udp_batch, max_packets)? {
            UdpRecvOutcome::Packets(count) => count,
            UdpRecvOutcome::WouldBlock => return Ok(None),
            UdpRecvOutcome::Error => {
                self.best_effort_reconcile();
                return Ok(None);
            }
        };
        Ok(Some(received_buffers))
    }

    fn handle_udp_batch(&mut self, received_buffers: usize) -> Result<usize> {
        let mut processed_packets = 0usize;

        for buffer_index in 0..received_buffers {
            let Some(meta) = self.udp_batch.metas.get(buffer_index).copied() else {
                continue;
            };
            let Some(buffer) = self.udp_batch.buffers.get(buffer_index) else {
                continue;
            };
            let available_len = meta.len.min(buffer.len());
            if meta.len > buffer.len() {
                self.counters.invalid_packet_drops += 1;
                tracing::warn!(
                    packet_len = meta.len,
                    buffer_len = buffer.len(),
                    "dropping truncated UDP receive batch buffer"
                );
                continue;
            }
            if available_len == 0 {
                self.handle_one_udp_payload(buffer_index, 0, 0)?;
                processed_packets += 1;
                continue;
            }

            let stride = udp_meta_stride(meta);
            let mut start = 0usize;
            while start < available_len {
                let end = start.saturating_add(stride).min(available_len);
                self.handle_one_udp_payload(buffer_index, start, end)?;
                processed_packets += 1;
                start = end;
            }
        }

        Ok(processed_packets)
    }

    fn handle_one_udp_payload(
        &mut self,
        buffer_index: usize,
        packet_start: usize,
        packet_end: usize,
    ) -> Result<()> {
        let packet_len = packet_end.saturating_sub(packet_start);
        self.counters.udp_rx_packets += 1;
        self.counters.udp_rx_bytes += packet_len as u64;

        let Some(dst) = ({
            let packet = &self.udp_batch.buffers[buffer_index][packet_start..packet_end];
            ipv6_destination(packet)
        }) else {
            self.counters.invalid_packet_drops += 1;
            tracing::debug!("dropping invalid UDP payload");
            return Ok(());
        };
        tracing::debug!(
            destination = %dst,
            bytes = packet_len,
            "dataplane parsed UDP payload destination"
        );

        if self.is_local(dst) {
            tracing::debug!(
                destination = %dst,
                bytes = packet_len,
                "dataplane delivering UDP payload to TUN"
            );
            let write_result = {
                let packet = &self.udp_batch.buffers[buffer_index][packet_start..packet_end];
                write_tun_packet(&self.tun_device, packet)
            };
            if self.handle_tun_write_result(write_result)? {
                self.counters.local_delivered_packets += 1;
            }
            return Ok(());
        }

        if !{
            let packet = &self.udp_batch.buffers[buffer_index][packet_start..packet_end];
            has_forwardable_hop_limit(packet)
        } {
            self.counters.hop_limit_drops += 1;
            tracing::debug!(destination = %dst, "dropping packet with exhausted hop limit");
            return Ok(());
        }

        let Some(route) = self.lookup_fast_route(dst) else {
            self.counters.no_route_drops += 1;
            tracing::debug!(destination = %dst, "dataplane has no route for forwarded UDP packet");
            return Ok(());
        };

        {
            let packet = &mut self.udp_batch.buffers[buffer_index][packet_start..packet_end];
            debug_assert!(decrement_hop_limit(packet));
        }

        tracing::debug!(
            destination = %dst,
            socket_slot = route.socket_slot,
            next_hop = %route.next_hop_ll,
            mtu = route.mtu,
            "dataplane resolved route for forwarded UDP packet"
        );
        let send_result = {
            let packet = &self.udp_batch.buffers[buffer_index][packet_start..packet_end];
            send_via_route(&self.sockets, self.udp_port, packet, route)
        };
        if self.handle_udp_send_result(send_result, route, "forwarding UDP packet")? {
            self.counters.udp_forwarded_packets += 1;
        }
        Ok(())
    }

    fn is_local(&self, addr: Ipv6Addr) -> bool {
        self.fib.locals.contains(&host_key(addr))
    }

    fn lookup_fast_route(&self, addr: Ipv6Addr) -> Option<FastFibEntry> {
        self.fast_routes.get(&host_key(addr)).copied()
    }

    fn handle_udp_send_result(
        &mut self,
        result: Result<PacketWriteOutcome>,
        route: FastFibEntry,
        context: &str,
    ) -> Result<bool> {
        match result {
            Ok(PacketWriteOutcome::Sent(sent)) => {
                self.counters.udp_tx_packets += 1;
                self.counters.udp_tx_bytes += sent as u64;
                Ok(true)
            }
            Ok(PacketWriteOutcome::WouldBlock) => {
                self.counters.udp_send_would_block_drops += 1;
                Ok(false)
            }
            Err(err) => {
                self.counters.udp_send_errors += 1;
                tracing::warn!(
                    socket_slot = route.socket_slot,
                    next_hop = %route.next_hop_ll,
                    context,
                    error = %err,
                    "failed to send dataplane packet via route"
                );
                self.best_effort_reconcile();
                Ok(false)
            }
        }
    }

    fn handle_tun_write_result(&mut self, result: Result<PacketWriteOutcome>) -> Result<bool> {
        match result {
            Ok(PacketWriteOutcome::Sent(sent)) => {
                self.counters.tun_tx_packets += 1;
                self.counters.tun_tx_bytes += sent as u64;
                Ok(true)
            }
            Ok(PacketWriteOutcome::WouldBlock) => {
                self.counters.tun_send_would_block_drops += 1;
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    fn apply_snapshot(&mut self, snapshot: Arc<FibSnapshot>) -> Result<()> {
        self.fib = snapshot;
        tracing::info!(
            locals = self.fib.locals.len(),
            admitted_interfaces = self.fib.admitted_interfaces.len(),
            routes = self.fib.routes.len(),
            "dataplane applied FIB snapshot"
        );
        self.reconcile_sockets()
    }

    fn reconcile_sockets(&mut self) -> Result<()> {
        self.last_reconcile_attempt = Instant::now();
        let current_ifindices = self.current_ifindices();
        let actions = plan_socket_actions(
            &current_ifindices,
            &self.fib.admitted_interfaces,
            |ifname| {
                if_nametoindex(ifname).wrap_err_with(|| format!("resolving ifindex for {}", ifname))
            },
        );
        let mut retry_needed = !actions.skipped.is_empty();

        for skipped in actions.skipped {
            tracing::warn!(
                interface = %skipped.ifname,
                reason = %skipped.reason,
                "skipping dataplane socket reconcile for interface"
            );
        }

        for action in actions.actions {
            match action {
                SocketAction::Add { ifname, ifindex } => {
                    if let Err(err) = self.add_socket(ifname.clone(), ifindex) {
                        retry_needed = true;
                        tracing::warn!(
                            interface = %ifname,
                            ifindex,
                            error = %err,
                            "failed to add dataplane socket for interface"
                        );
                    }
                }
                SocketAction::Refresh { ifname, ifindex } => {
                    if let Err(err) = self.refresh_socket(&ifname, ifindex) {
                        retry_needed = true;
                        tracing::warn!(
                            interface = %ifname,
                            ifindex,
                            error = %err,
                            "failed to refresh dataplane socket for interface"
                        );
                    }
                }
                SocketAction::Remove { ifname } => {
                    if let Err(err) = self.remove_socket(&ifname) {
                        tracing::warn!(
                            interface = %ifname,
                            error = %err,
                            "failed to remove dataplane socket for interface"
                        );
                    }
                }
            }
        }

        retry_needed |=
            has_missing_admitted_sockets(&self.ifname_to_slab, &self.fib.admitted_interfaces);
        self.needs_reconcile_retry = retry_needed;
        self.rebuild_fast_routes();
        Ok(())
    }

    fn best_effort_reconcile(&mut self) {
        if let Err(err) = self.reconcile_sockets() {
            tracing::warn!(error = %err, "dataplane socket reconcile failed");
        }
    }

    fn maybe_retry_reconcile(&mut self) {
        if !self.needs_reconcile_retry {
            return;
        }
        if self.last_reconcile_attempt.elapsed() < RECONCILE_RETRY_INTERVAL {
            return;
        }
        tracing::debug!(
            admitted_interfaces = self.fib.admitted_interfaces.len(),
            current_sockets = self.ifname_to_slab.len(),
            "retrying dataplane socket reconcile after earlier skip/failure"
        );
        self.best_effort_reconcile();
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
        let slab_key = self.insert_registered_socket(ifname.clone(), ifindex, socket)?;
        self.ifname_to_slab.insert(ifname, slab_key);
        Ok(())
    }

    fn refresh_socket(&mut self, ifname: &str, ifindex: u32) -> Result<()> {
        let Some(old_slab_key) = self.ifname_to_slab.get(ifname).copied() else {
            return self.add_socket(ifname.into(), ifindex);
        };

        let new_ifname: Box<str> = ifname.into();
        let new_socket = open_udp_socket(self.udp_port, ifname, ifindex)
            .wrap_err_with(|| format!("opening replacement UDP socket for {}", ifname))?;
        let new_slab_key =
            self.insert_registered_socket(new_ifname.clone(), ifindex, new_socket)?;
        let previous = self.ifname_to_slab.insert(new_ifname.clone(), new_slab_key);
        debug_assert_eq!(previous, Some(old_slab_key));

        let Some(mut old_socket) = self.sockets.try_remove(old_slab_key) else {
            return Ok(());
        };
        if let Err(err) = self.poll.registry().deregister(&mut old_socket.socket) {
            tracing::warn!(
                interface = %ifname,
                error = %err,
                "failed to deregister old dataplane socket after refresh"
            );
        }
        Ok(())
    }

    fn insert_registered_socket(
        &mut self,
        ifname: Box<str>,
        ifindex: u32,
        mut socket: MioUdpSocket,
    ) -> Result<usize> {
        let udp_state = UdpSocketState::new(UdpSockRef::from(&socket))
            .wrap_err_with(|| format!("initializing UDP fast-path state for {}", ifname))?;
        let entry = self.sockets.vacant_entry();
        let slab_key = entry.key();
        let token = Token(slab_key + 1);
        self.poll
            .registry()
            .register(&mut socket, token, Interest::READABLE)
            .wrap_err_with(|| format!("registering UDP socket for {}", ifname))?;
        entry.insert(InterfaceSocket {
            ifname,
            ifindex,
            socket,
            udp_state,
        });
        Ok(slab_key)
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

    fn rebuild_fast_routes(&mut self) {
        self.fast_routes = compile_fast_routes(self.fib.as_ref(), &self.ifname_to_slab);
        let skipped_routes = self.fib.routes.len().saturating_sub(self.fast_routes.len());
        tracing::info!(
            compiled_routes = self.fast_routes.len(),
            skipped_missing_socket = skipped_routes,
            sockets = self.ifname_to_slab.len(),
            "dataplane rebuilt fast FIB"
        );
    }

    fn maybe_log_counters(&mut self) {
        if self.last_counter_log.elapsed() < COUNTER_LOG_INTERVAL {
            return;
        }

        let delta = self.counters.saturating_delta(self.last_logged_counters);
        self.last_counter_log = Instant::now();
        self.last_logged_counters = self.counters;

        if !delta.has_activity() {
            return;
        }

        tracing::info!(
            tun_rx_packets_delta = delta.tun_rx_packets,
            tun_rx_bytes_delta = delta.tun_rx_bytes,
            udp_rx_packets_delta = delta.udp_rx_packets,
            udp_rx_bytes_delta = delta.udp_rx_bytes,
            udp_tx_packets_delta = delta.udp_tx_packets,
            udp_tx_bytes_delta = delta.udp_tx_bytes,
            tun_tx_packets_delta = delta.tun_tx_packets,
            tun_tx_bytes_delta = delta.tun_tx_bytes,
            tun_to_udp_packets_delta = delta.tun_to_udp_packets,
            udp_forwarded_packets_delta = delta.udp_forwarded_packets,
            local_delivered_packets_delta = delta.local_delivered_packets,
            no_route_drops_delta = delta.no_route_drops,
            invalid_packet_drops_delta = delta.invalid_packet_drops,
            self_directed_drops_delta = delta.self_directed_drops,
            hop_limit_drops_delta = delta.hop_limit_drops,
            udp_send_would_block_drops_delta = delta.udp_send_would_block_drops,
            tun_send_would_block_drops_delta = delta.tun_send_would_block_drops,
            udp_send_errors_delta = delta.udp_send_errors,
            tun_rx_packets_total = self.counters.tun_rx_packets,
            udp_rx_packets_total = self.counters.udp_rx_packets,
            udp_tx_packets_total = self.counters.udp_tx_packets,
            tun_tx_packets_total = self.counters.tun_tx_packets,
            no_route_drops_total = self.counters.no_route_drops,
            udp_send_would_block_drops_total = self.counters.udp_send_would_block_drops,
            tun_send_would_block_drops_total = self.counters.tun_send_would_block_drops,
            udp_send_errors_total = self.counters.udp_send_errors,
            "dataplane counters"
        );
    }
}

fn recv_udp_packets(
    socket: &InterfaceSocket,
    batch: &mut UdpRecvBatch,
    max_packets: usize,
) -> Result<UdpRecvOutcome> {
    match batch.recv_from(socket, max_packets) {
        Ok(received_buffers) if received_buffers > 0 => {
            tracing::debug!(
                interface = %socket.ifname,
                received_buffers,
                "dataplane received UDP packet batch"
            );
            Ok(UdpRecvOutcome::Packets(received_buffers))
        }
        Ok(_) => Ok(UdpRecvOutcome::WouldBlock),
        Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(UdpRecvOutcome::WouldBlock),
        Err(err) => {
            tracing::warn!(
                interface = %socket.ifname,
                error = %err,
                "failed to receive UDP packet batch on dataplane socket"
            );
            Ok(UdpRecvOutcome::Error)
        }
    }
}

fn send_via_route(
    sockets: &Slab<InterfaceSocket>,
    udp_port: u16,
    packet: &[u8],
    route: FastFibEntry,
) -> Result<PacketWriteOutcome> {
    let Some(socket) = sockets.get(route.socket_slot) else {
        return Err(eyre!("stale UDP socket slot {}", route.socket_slot));
    };

    let peer = SocketAddr::V6(SocketAddrV6::new(
        route.next_hop_ll,
        udp_port,
        0,
        socket.ifindex,
    ));
    tracing::debug!(
        interface = %socket.ifname,
        ifindex = socket.ifindex,
        peer = %peer,
        bytes = packet.len(),
        "dataplane sending UDP packet"
    );
    let transmit = Transmit {
        destination: peer,
        ecn: None,
        contents: packet,
        segment_size: None,
        src_ip: None,
    };
    match socket.socket.try_io(|| {
        socket
            .udp_state
            .try_send(UdpSockRef::from(&socket.socket), &transmit)
    }) {
        Ok(()) => {
            tracing::debug!(
                interface = %socket.ifname,
                ifindex = socket.ifindex,
                peer = %peer,
                bytes = packet.len(),
                "dataplane sent UDP packet"
            );
            Ok(PacketWriteOutcome::Sent(packet.len()))
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => {
            tracing::debug!(
                interface = %socket.ifname,
                ifindex = socket.ifindex,
                peer = %peer,
                bytes = packet.len(),
                "dropping dataplane packet because UDP socket send would block"
            );
            Ok(PacketWriteOutcome::WouldBlock)
        }
        Err(err) => Err(err).wrap_err_with(|| format!("sending packet via {}", socket.ifname)),
    }
}

fn write_tun_packet(tun_device: &SyncDevice, packet: &[u8]) -> Result<PacketWriteOutcome> {
    match tun_device.send(packet) {
        Ok(sent) => Ok(PacketWriteOutcome::Sent(sent)),
        Err(err) if err.kind() == ErrorKind::WouldBlock => {
            tracing::debug!(
                bytes = packet.len(),
                "dropping dataplane packet because TUN reinjection would block"
            );
            Ok(PacketWriteOutcome::WouldBlock)
        }
        Err(err) => Err(err).wrap_err("writing inner packet to TUN"),
    }
}

fn compile_fast_routes<S>(
    fib: &FibSnapshot,
    ifname_to_slab: &HashMap<Box<str>, usize, S>,
) -> HashMap<HostKey, FastFibEntry, RandomState>
where
    S: BuildHasher,
{
    let mut routes = HashMap::with_capacity_and_hasher(fib.routes.len(), RandomState::new());
    for (dst, route) in &fib.routes {
        let Some(&socket_slot) = ifname_to_slab.get(route.ifname.as_ref()) else {
            continue;
        };
        routes.insert(
            *dst,
            FastFibEntry {
                next_hop_ll: route.next_hop_ll,
                socket_slot,
                mtu: route.mtu,
            },
        );
    }
    routes
}

fn plan_socket_actions<F, S1, S2, E>(
    current_ifindices: &HashMap<Box<str>, u32, S1>,
    admitted_interfaces: &HashSet<Box<str>, S2>,
    mut resolve_ifindex: F,
) -> SocketPlan
where
    F: FnMut(&str) -> std::result::Result<u32, E>,
    S1: BuildHasher,
    S2: BuildHasher,
    E: std::fmt::Display,
{
    let mut actions = Vec::new();
    let mut skipped = Vec::new();

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
        let ifindex = match resolve_ifindex(ifname.as_ref()) {
            Ok(ifindex) => ifindex,
            Err(err) => {
                skipped.push(SkippedInterface {
                    ifname,
                    reason: err.to_string(),
                });
                continue;
            }
        };
        match current_ifindices.get(ifname.as_ref()).copied() {
            None => actions.push(SocketAction::Add { ifname, ifindex }),
            Some(existing_ifindex) if existing_ifindex != ifindex => {
                actions.push(SocketAction::Refresh { ifname, ifindex });
            }
            Some(_) => {}
        }
    }

    SocketPlan { actions, skipped }
}

fn has_missing_admitted_sockets<S1, S2>(
    current_ifnames: &HashMap<Box<str>, usize, S1>,
    admitted_interfaces: &HashSet<Box<str>, S2>,
) -> bool
where
    S1: BuildHasher,
    S2: BuildHasher,
{
    admitted_interfaces
        .iter()
        .any(|ifname| !current_ifnames.contains_key(ifname.as_ref()))
}

fn open_udp_socket(port: u16, ifname: &str, ifindex: u32) -> io::Result<MioUdpSocket> {
    let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_only_v6(true)?;
    socket.set_nonblocking(true)?;
    socket.set_recv_buffer_size(UDP_SOCKET_BUFFER_BYTES)?;
    socket.set_send_buffer_size(UDP_SOCKET_BUFFER_BYTES)?;

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

fn has_forwardable_hop_limit(packet: &[u8]) -> bool {
    if packet.len() < 40 || packet[0] >> 4 != 6 {
        return false;
    }
    packet[7] > 1
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv6Addr, UdpSocket as StdUdpSocket};
    use std::time::Duration;

    use ahash::RandomState;
    use hashbrown::{HashMap, HashSet};
    use mio::net::UdpSocket as MioUdpSocket;

    use crate::fib::{FibEntry, FibSnapshot, host_key};

    use super::{
        BATCH_SIZE, FastFibEntry, InterfaceSocket, RecvMeta, SocketAction, UdpRecvBatch,
        UdpRecvOutcome, UdpSockRef, UdpSocketState, compile_fast_routes, decrement_hop_limit,
        has_missing_admitted_sockets, ipv6_destination, plan_socket_actions, recv_udp_packets,
        udp_meta_stride,
    };

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
    fn udp_meta_stride_falls_back_to_single_datagram() {
        let mut meta = RecvMeta::default();
        meta.len = 1452;
        meta.stride = 0;
        assert_eq!(udp_meta_stride(meta), 1452);

        meta.stride = 4096;
        assert_eq!(udp_meta_stride(meta), 1452);

        meta.stride = 484;
        assert_eq!(udp_meta_stride(meta), 484);
    }

    #[test]
    fn udp_batch_recv_returns_single_datagram_without_full_batch() {
        let receiver = StdUdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).unwrap();
        let receiver_addr = receiver.local_addr().unwrap();
        let receiver = MioUdpSocket::from_std(receiver);
        let udp_state = UdpSocketState::new(UdpSockRef::from(&receiver)).unwrap();
        let socket = InterfaceSocket {
            ifname: "lo0".into(),
            ifindex: 0,
            socket: receiver,
            udp_state,
        };

        let sender = StdUdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).unwrap();
        let payload = sample_ipv6_packet("fde0::1234".parse().unwrap(), 32);
        sender.send_to(&payload, receiver_addr).unwrap();

        let mut batch = UdpRecvBatch::new();
        let started = std::time::Instant::now();
        let outcome = recv_udp_packets(&socket, &mut batch, BATCH_SIZE).unwrap();

        assert!(
            started.elapsed() < Duration::from_secs(1),
            "batched UDP receive should not wait for a full batch"
        );
        assert_eq!(outcome, UdpRecvOutcome::Packets(1));
        assert_eq!(batch.metas[0].len, payload.len());
        assert_eq!(&batch.buffers[0][..payload.len()], payload.as_slice());
    }

    #[test]
    fn socket_plan_uses_admitted_interfaces_not_routes() {
        let current_ifindices = HashMap::with_hasher(RandomState::new());
        let admitted_interfaces: HashSet<Box<str>> = ["mesh0".into()].into_iter().collect();

        let plan = plan_socket_actions(&current_ifindices, &admitted_interfaces, |ifname| {
            Ok::<_, &'static str>(match ifname {
                "mesh0" => 7,
                _ => unreachable!(),
            })
        });

        assert!(plan.skipped.is_empty());
        assert_eq!(
            plan.actions,
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
        let plan = plan_socket_actions(&current_ifindices, &admitted_interfaces, |ifname| {
            Ok::<_, &'static str>(match ifname {
                "mesh0" => 7,
                _ => unreachable!(),
            })
        });

        assert!(plan.skipped.is_empty());
        assert_eq!(
            plan.actions,
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

    #[test]
    fn socket_plan_skips_temporarily_unresolved_interfaces() {
        let current_ifindices = HashMap::with_hasher(RandomState::new());
        let admitted_interfaces: HashSet<Box<str>> =
            ["mesh0".into(), "mesh1".into()].into_iter().collect();

        let plan =
            plan_socket_actions(
                &current_ifindices,
                &admitted_interfaces,
                |ifname| match ifname {
                    "mesh0" => Ok(7),
                    "mesh1" => Err("temporary resolution failure"),
                    _ => unreachable!(),
                },
            );

        assert_eq!(
            plan.actions,
            vec![SocketAction::Add {
                ifname: "mesh0".into(),
                ifindex: 7,
            }]
        );
        assert_eq!(plan.skipped.len(), 1);
        assert_eq!(plan.skipped[0].ifname.as_ref(), "mesh1");
        assert_eq!(plan.skipped[0].reason, "temporary resolution failure");
    }

    #[test]
    fn detects_missing_admitted_socket_entries() {
        let mut current_ifnames = HashMap::with_hasher(RandomState::new());
        current_ifnames.insert("mesh0".into(), 1usize);

        let admitted_interfaces: HashSet<Box<str>> =
            ["mesh0".into(), "mesh1".into()].into_iter().collect();
        assert!(has_missing_admitted_sockets(
            &current_ifnames,
            &admitted_interfaces
        ));

        current_ifnames.insert("mesh1".into(), 2usize);
        assert!(!has_missing_admitted_sockets(
            &current_ifnames,
            &admitted_interfaces
        ));
    }

    #[test]
    fn compiled_routes_store_socket_slots_and_skip_missing_sockets() {
        let dst_with_socket = "fde0::1234".parse::<Ipv6Addr>().unwrap();
        let dst_without_socket = "fde0::5678".parse::<Ipv6Addr>().unwrap();

        let mut fib_routes = HashMap::with_hasher(RandomState::new());
        fib_routes.insert(
            host_key(dst_with_socket),
            FibEntry {
                next_hop_ll: "fe80::1".parse().unwrap(),
                ifname: "en2".into(),
                mtu: 1452,
            },
        );
        fib_routes.insert(
            host_key(dst_without_socket),
            FibEntry {
                next_hop_ll: "fe80::2".parse().unwrap(),
                ifname: "en3".into(),
                mtu: 1452,
            },
        );

        let fib = FibSnapshot {
            locals: HashSet::with_hasher(RandomState::new()),
            admitted_interfaces: ["en2".into(), "en3".into()].into_iter().collect(),
            routes: fib_routes,
        };
        let mut ifname_to_slab = HashMap::with_hasher(RandomState::new());
        ifname_to_slab.insert("en2".into(), 7usize);

        let fast_routes = compile_fast_routes(&fib, &ifname_to_slab);

        assert_eq!(fast_routes.len(), 1);
        assert_eq!(
            fast_routes.get(&host_key(dst_with_socket)),
            Some(&FastFibEntry {
                next_hop_ll: "fe80::1".parse().unwrap(),
                socket_slot: 7,
                mtu: 1452,
            })
        );
        assert!(!fast_routes.contains_key(&host_key(dst_without_socket)));
    }
}
