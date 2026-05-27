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

use std::collections::VecDeque;
use std::hash::BuildHasher;
use std::io::{self, ErrorKind, IoSliceMut, Write};
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, TcpListener, TcpStream, UdpSocket};
use std::num::NonZeroU32;
use std::ops::Range;
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
use mio::net::{
    TcpListener as MioTcpListener, TcpStream as MioTcpStream, UdpSocket as MioUdpSocket,
};
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use nix::net::if_::if_nametoindex;
use slab::Slab;
use socket2::{Domain, Protocol, Socket, Type};
use tokio::sync::oneshot;
use tun_rs::SyncDevice;

use crate::config::{TCP_PENDING_LIMIT_BYTES, TransportMode, UDP_TUN_MTU};
use crate::fib::{FibSnapshot, HostKey, host_key};

const TOKEN_TUN: Token = Token(0);
const TOKEN_INTERFACE_BASE: usize = 1;
const TOKEN_TCP_STREAM_BASE: usize = 1_000_000;
const MAX_POLL_EVENTS: usize = 128;
const POLL_INTERVAL: Duration = Duration::from_millis(250);
const RECONCILE_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const COUNTER_LOG_INTERVAL: Duration = Duration::from_secs(5);
const UDP_SOCKET_BUFFER_BYTES: usize = 4 * 1024 * 1024;
const TUN_DRAIN_BUDGET: usize = 64;
const UDP_DRAIN_BUDGET: usize = 64;
const MAX_GRO_SEGMENTS: usize = 64;
const TCP_READ_BUFFER_BYTES: usize = 256 * 1024;
const TCP_ACCEPT_DRAIN_BUDGET: usize = 64;
const TCP_STREAM_READ_DRAIN_BUDGET: usize = 64;

pub struct DataplaneConfig {
    pub tun_device: Arc<SyncDevice>,
    pub udp_port: u16,
    pub transport_mode: TransportMode,
    pub tun_mtu: u16,
    pub tcp_batch_target_bytes: usize,
    pub tcp_socket_buffer_bytes: usize,
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
    transport_mode: TransportMode,
    fib: Arc<FibSnapshot>,
    fast_routes: HashMap<HostKey, FastFibEntry, RandomState>,
    fib_updates: Receiver<Arc<FibSnapshot>>,
    stop_recv: Receiver<()>,
    poll: Poll,
    events: Events,
    tun_mtu: usize,
    tcp_batch_target_bytes: usize,
    tcp_socket_buffer_bytes: usize,
    sockets: Slab<InterfaceSocket>,
    ifname_to_slab: HashMap<Box<str>, usize>,
    tcp_streams: Slab<TcpPeerStream>,
    tcp_peer_to_stream: HashMap<TcpPeerKey, usize>,
    needs_reconcile_retry: bool,
    last_reconcile_attempt: Instant,
    counters: DataplaneCounters,
    last_counter_log: Instant,
    last_logged_counters: DataplaneCounters,
    tun_buf: Vec<u8>,
    udp_batch: UdpRecvBatch,
}

struct InterfaceSocket {
    ifname: Box<str>,
    ifindex: u32,
    io: InterfaceIo,
}

enum InterfaceIo {
    Udp {
        socket: MioUdpSocket,
        udp_state: UdpSocketState,
    },
    #[cfg(any(not(target_os = "linux"), test))]
    TcpInterface,
    TcpListener {
        listener: MioTcpListener,
    },
}

struct TcpPeerStream {
    interface_slot: usize,
    peer_key: Option<TcpPeerKey>,
    peer: SocketAddr,
    stream: MioTcpStream,
    connected: bool,
    write_interest: bool,
    read_buf: TcpFrameDecoder,
    write_buf: TcpWriteBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TcpPeerKey {
    socket_slot: usize,
    next_hop_ll: Ipv6Addr,
}

#[derive(Debug)]
struct TcpFrameDecoder {
    bytes: Vec<u8>,
    offset: usize,
}

#[derive(Debug)]
struct TcpWriteBuffer {
    bytes: Vec<u8>,
    offset: usize,
    frame_ends: VecDeque<TcpQueuedFrame>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct PacketSizeHistogram {
    le_16k: u64,
    le_32k: u64,
    le_48k: u64,
    le_60k: u64,
    le_65k: u64,
    gt_65k: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcpFrameOrigin {
    TunIngress,
    Forwarded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TcpQueuedFrame {
    end_offset: usize,
    origin: TcpFrameOrigin,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct TcpCompletedFrames {
    total: usize,
    tun_ingress: usize,
    forwarded: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcpFrameError {
    ZeroLength,
    Oversized { len: usize, max: usize },
}

#[derive(Debug, Clone, Copy)]
struct ReadyEvent {
    token: Token,
    readable: bool,
    writable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcpQueueOutcome {
    Queued,
    Full,
    Oversized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcpConnectOutcome {
    Connected,
    InProgress,
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
    tun_rx_sizes: PacketSizeHistogram,
    udp_rx_packets: u64,
    udp_rx_bytes: u64,
    udp_tx_packets: u64,
    udp_tx_bytes: u64,
    tun_tx_packets: u64,
    tun_tx_bytes: u64,
    tun_tx_sizes: PacketSizeHistogram,
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
    tcp_connects: u64,
    tcp_accepts: u64,
    tcp_rejected_peers: u64,
    tcp_reconnects: u64,
    tcp_queued_packets: u64,
    tcp_tx_batches: u64,
    tcp_written_frames: u64,
    tcp_tx_bytes: u64,
    tcp_reregisters: u64,
    tcp_rx_batches: u64,
    tcp_rx_frames: u64,
    tcp_rx_bytes: u64,
    tcp_partial_writes: u64,
    tcp_blocked_writes: u64,
    tcp_queue_drops: u64,
    tcp_frame_errors: u64,
    tcp_stream_errors: u64,
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

impl PacketSizeHistogram {
    fn observe(&mut self, len: usize) {
        match len {
            0..=16_384 => self.le_16k += 1,
            16_385..=32_768 => self.le_32k += 1,
            32_769..=49_152 => self.le_48k += 1,
            49_153..=61_440 => self.le_60k += 1,
            61_441..=65_535 => self.le_65k += 1,
            _ => self.gt_65k += 1,
        }
    }

    fn saturating_delta(self, previous: Self) -> Self {
        Self {
            le_16k: self.le_16k.saturating_sub(previous.le_16k),
            le_32k: self.le_32k.saturating_sub(previous.le_32k),
            le_48k: self.le_48k.saturating_sub(previous.le_48k),
            le_60k: self.le_60k.saturating_sub(previous.le_60k),
            le_65k: self.le_65k.saturating_sub(previous.le_65k),
            gt_65k: self.gt_65k.saturating_sub(previous.gt_65k),
        }
    }
}

impl DataplaneCounters {
    fn saturating_delta(self, previous: Self) -> Self {
        Self {
            tun_rx_packets: self.tun_rx_packets.saturating_sub(previous.tun_rx_packets),
            tun_rx_bytes: self.tun_rx_bytes.saturating_sub(previous.tun_rx_bytes),
            tun_rx_sizes: self.tun_rx_sizes.saturating_delta(previous.tun_rx_sizes),
            udp_rx_packets: self.udp_rx_packets.saturating_sub(previous.udp_rx_packets),
            udp_rx_bytes: self.udp_rx_bytes.saturating_sub(previous.udp_rx_bytes),
            udp_tx_packets: self.udp_tx_packets.saturating_sub(previous.udp_tx_packets),
            udp_tx_bytes: self.udp_tx_bytes.saturating_sub(previous.udp_tx_bytes),
            tun_tx_packets: self.tun_tx_packets.saturating_sub(previous.tun_tx_packets),
            tun_tx_bytes: self.tun_tx_bytes.saturating_sub(previous.tun_tx_bytes),
            tun_tx_sizes: self.tun_tx_sizes.saturating_delta(previous.tun_tx_sizes),
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
            tcp_connects: self.tcp_connects.saturating_sub(previous.tcp_connects),
            tcp_accepts: self.tcp_accepts.saturating_sub(previous.tcp_accepts),
            tcp_rejected_peers: self
                .tcp_rejected_peers
                .saturating_sub(previous.tcp_rejected_peers),
            tcp_reconnects: self.tcp_reconnects.saturating_sub(previous.tcp_reconnects),
            tcp_queued_packets: self
                .tcp_queued_packets
                .saturating_sub(previous.tcp_queued_packets),
            tcp_tx_batches: self.tcp_tx_batches.saturating_sub(previous.tcp_tx_batches),
            tcp_written_frames: self
                .tcp_written_frames
                .saturating_sub(previous.tcp_written_frames),
            tcp_tx_bytes: self.tcp_tx_bytes.saturating_sub(previous.tcp_tx_bytes),
            tcp_reregisters: self
                .tcp_reregisters
                .saturating_sub(previous.tcp_reregisters),
            tcp_rx_batches: self.tcp_rx_batches.saturating_sub(previous.tcp_rx_batches),
            tcp_rx_frames: self.tcp_rx_frames.saturating_sub(previous.tcp_rx_frames),
            tcp_rx_bytes: self.tcp_rx_bytes.saturating_sub(previous.tcp_rx_bytes),
            tcp_partial_writes: self
                .tcp_partial_writes
                .saturating_sub(previous.tcp_partial_writes),
            tcp_blocked_writes: self
                .tcp_blocked_writes
                .saturating_sub(previous.tcp_blocked_writes),
            tcp_queue_drops: self
                .tcp_queue_drops
                .saturating_sub(previous.tcp_queue_drops),
            tcp_frame_errors: self
                .tcp_frame_errors
                .saturating_sub(previous.tcp_frame_errors),
            tcp_stream_errors: self
                .tcp_stream_errors
                .saturating_sub(previous.tcp_stream_errors),
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
            || self.tcp_connects != 0
            || self.tcp_accepts != 0
            || self.tcp_rejected_peers != 0
            || self.tcp_reconnects != 0
            || self.tcp_queued_packets != 0
            || self.tcp_tx_batches != 0
            || self.tcp_written_frames != 0
            || self.tcp_reregisters != 0
            || self.tcp_rx_batches != 0
            || self.tcp_rx_frames != 0
            || self.tcp_blocked_writes != 0
            || self.tcp_queue_drops != 0
            || self.tcp_frame_errors != 0
            || self.tcp_stream_errors != 0
    }
}

impl UdpRecvBatch {
    fn new(packet_buffer_bytes: usize) -> Self {
        let recv_buffer_bytes = packet_buffer_bytes.saturating_mul(MAX_GRO_SEGMENTS).max(1);
        let buffers = (0..BATCH_SIZE)
            .map(|_| vec![0u8; recv_buffer_bytes].into_boxed_slice())
            .collect();
        let metas = vec![RecvMeta::default(); BATCH_SIZE];
        Self { buffers, metas }
    }

    fn recv_from(
        &mut self,
        socket: &MioUdpSocket,
        udp_state: &UdpSocketState,
        max_packets: usize,
    ) -> io::Result<usize> {
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

        socket.try_io(|| {
            udp_state.recv(
                UdpSockRef::from(socket),
                slices.as_mut_slice(),
                &mut self.metas[..batch_len],
            )
        })
    }
}

impl TcpFrameDecoder {
    fn new(read_buffer_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(read_buffer_bytes),
            offset: 0,
        }
    }

    fn read_from(&mut self, stream: &MioTcpStream, read_buffer_bytes: usize) -> io::Result<usize> {
        self.compact_if_needed(read_buffer_bytes);
        self.bytes.reserve(read_buffer_bytes);
        let start = self.bytes.len();
        let spare = self.bytes.spare_capacity_mut();
        let read_len = spare.len().min(read_buffer_bytes);
        if read_len == 0 {
            return Ok(0);
        }

        loop {
            // SAFETY: `ptr` points at spare Vec capacity. `read(2)` writes at
            // most `read_len` bytes, and the Vec length is extended only by the
            // number of bytes the kernel reports as initialized.
            let ret = unsafe {
                libc::read(
                    stream.as_raw_fd(),
                    spare.as_ptr().cast::<libc::c_void>().cast_mut(),
                    read_len,
                )
            };
            if ret >= 0 {
                let read_len = ret as usize;
                // SAFETY: the kernel initialized exactly `read_len` bytes in
                // the spare capacity on success.
                unsafe {
                    self.bytes.set_len(start + read_len);
                }
                return Ok(read_len);
            }

            let err = io::Error::last_os_error();
            if err.kind() != ErrorKind::Interrupted {
                return Err(err);
            }
        }
    }

    #[cfg(test)]
    fn push(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn next_frame_range(
        &mut self,
        max_packet_len: usize,
    ) -> std::result::Result<Option<Range<usize>>, TcpFrameError> {
        if self.bytes.len().saturating_sub(self.offset) < 2 {
            self.compact_if_empty();
            return Ok(None);
        }
        let len =
            u16::from_be_bytes([self.bytes[self.offset], self.bytes[self.offset + 1]]) as usize;
        if len == 0 {
            return Err(TcpFrameError::ZeroLength);
        }
        if len > max_packet_len {
            return Err(TcpFrameError::Oversized {
                len,
                max: max_packet_len,
            });
        }
        let frame_start = self.offset + 2;
        let frame_end = frame_start + len;
        if self.bytes.len() < frame_end {
            return Ok(None);
        }
        self.offset = frame_end;
        Ok(Some(frame_start..frame_end))
    }

    fn finish_frame(&mut self, compact_threshold: usize) {
        self.compact_if_needed(compact_threshold);
    }

    fn compact_if_empty(&mut self) {
        if self.offset == self.bytes.len() {
            self.bytes.clear();
            self.offset = 0;
        }
    }

    fn compact_if_needed(&mut self, compact_threshold: usize) {
        if self.offset == self.bytes.len() {
            self.bytes.clear();
            self.offset = 0;
        } else if self.offset >= compact_threshold {
            self.bytes.drain(..self.offset);
            self.offset = 0;
        }
    }
}

impl Default for TcpFrameDecoder {
    fn default() -> Self {
        Self::new(TCP_READ_BUFFER_BYTES)
    }
}

impl TcpWriteBuffer {
    fn new(batch_target_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(batch_target_bytes),
            offset: 0,
            frame_ends: VecDeque::new(),
        }
    }

    fn queue_frame(
        &mut self,
        packet: &[u8],
        origin: TcpFrameOrigin,
        pending_limit_bytes: usize,
    ) -> TcpQueueOutcome {
        let Ok(packet_len) = u16::try_from(packet.len()) else {
            return TcpQueueOutcome::Oversized;
        };
        if self.pending_len() + 2 + packet.len() > pending_limit_bytes {
            return TcpQueueOutcome::Full;
        }
        self.bytes.extend_from_slice(&packet_len.to_be_bytes());
        self.bytes.extend_from_slice(packet);
        self.frame_ends.push_back(TcpQueuedFrame {
            end_offset: self.bytes.len(),
            origin,
        });
        TcpQueueOutcome::Queued
    }

    fn pending(&self) -> &[u8] {
        &self.bytes[self.offset..]
    }

    fn pending_len(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn has_pending(&self) -> bool {
        self.pending_len() != 0
    }

    fn consume(&mut self, bytes: usize, compact_threshold: usize) -> TcpCompletedFrames {
        self.offset = self.offset.saturating_add(bytes).min(self.bytes.len());
        let mut completed = TcpCompletedFrames::default();
        while self
            .frame_ends
            .front()
            .is_some_and(|frame| frame.end_offset <= self.offset)
        {
            let frame = self.frame_ends.pop_front().expect("front checked");
            completed.total += 1;
            match frame.origin {
                TcpFrameOrigin::TunIngress => completed.tun_ingress += 1,
                TcpFrameOrigin::Forwarded => completed.forwarded += 1,
            }
        }
        if self.offset == self.bytes.len() {
            self.clear();
        } else if self.offset >= compact_threshold {
            let drained = self.offset;
            self.bytes.drain(..self.offset);
            for frame in &mut self.frame_ends {
                frame.end_offset = frame.end_offset.saturating_sub(drained);
            }
            self.offset = 0;
        }
        completed
    }

    fn clear(&mut self) {
        self.bytes.clear();
        self.offset = 0;
        self.frame_ends.clear();
    }
}

fn udp_meta_stride(meta: RecvMeta) -> usize {
    if meta.stride == 0 || meta.stride > meta.len {
        meta.len.max(1)
    } else {
        meta.stride
    }
}

fn interface_token(slab_key: usize) -> Token {
    Token(TOKEN_INTERFACE_BASE + slab_key)
}

fn tcp_stream_token(stream_key: usize) -> Token {
    Token(TOKEN_TCP_STREAM_BASE + stream_key)
}

fn token_to_interface(token: Token) -> Option<usize> {
    let raw = token.0;
    if (TOKEN_INTERFACE_BASE..TOKEN_TCP_STREAM_BASE).contains(&raw) {
        Some(raw - TOKEN_INTERFACE_BASE)
    } else {
        None
    }
}

fn token_to_tcp_stream(token: Token) -> Option<usize> {
    token
        .0
        .checked_sub(TOKEN_TCP_STREAM_BASE)
        .filter(|_| token.0 >= TOKEN_TCP_STREAM_BASE)
}

fn udp_batch_packet_bytes(transport_mode: TransportMode, tun_mtu: u16) -> usize {
    match transport_mode {
        TransportMode::Udp => usize::from(tun_mtu),
        TransportMode::Tcp => usize::from(UDP_TUN_MTU),
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
            transport_mode: config.transport_mode,
            fib: config.initial_fib,
            fast_routes: HashMap::with_hasher(RandomState::new()),
            fib_updates,
            stop_recv,
            poll,
            events,
            tun_mtu: usize::from(config.tun_mtu),
            tcp_batch_target_bytes: config.tcp_batch_target_bytes,
            tcp_socket_buffer_bytes: config.tcp_socket_buffer_bytes,
            sockets: Slab::new(),
            ifname_to_slab: HashMap::new(),
            tcp_streams: Slab::new(),
            tcp_peer_to_stream: HashMap::new(),
            needs_reconcile_retry: false,
            last_reconcile_attempt: Instant::now(),
            counters: DataplaneCounters::default(),
            last_counter_log: Instant::now(),
            last_logged_counters: DataplaneCounters::default(),
            tun_buf: vec![0u8; usize::from(config.tun_mtu)],
            udp_batch: UdpRecvBatch::new(udp_batch_packet_bytes(
                config.transport_mode,
                config.tun_mtu,
            )),
        };
        worker.reconcile_sockets()?;
        Ok(worker)
    }

    fn run(&mut self) -> Result<()> {
        loop {
            if self.stop_requested()? {
                self.flush_tcp_writes("stopping dataplane");
                return Ok(());
            }

            self.drain_snapshot_updates()?;
            self.maybe_retry_reconcile();

            self.poll
                .poll(&mut self.events, Some(POLL_INTERVAL))
                .wrap_err("polling dataplane fds")?;

            let mut ready = ArrayVec::<ReadyEvent, MAX_POLL_EVENTS>::new();
            for event in self.events.iter() {
                let ready_event = ReadyEvent {
                    token: event.token(),
                    readable: event.is_readable(),
                    writable: event.is_writable(),
                };
                if ready.try_push(ready_event).is_err() {
                    break;
                }
            }
            for event in ready {
                match event.token {
                    TOKEN_TUN => {
                        tracing::trace!("dataplane poll signaled TUN readable");
                        self.drain_tun_ready()?
                    }
                    token if token_to_tcp_stream(token).is_some() => {
                        let stream_key = token_to_tcp_stream(token).expect("checked stream token");
                        self.drain_tcp_stream_ready(stream_key, event.readable, event.writable)?;
                    }
                    token if token_to_interface(token).is_some() => {
                        let slab_key = token_to_interface(token).expect("checked interface token");
                        self.drain_interface_ready(slab_key)?;
                    }
                    token => return Err(eyre!("invalid dataplane token {:?}", token)),
                }
            }
            self.flush_tcp_writes("dataplane poll loop boundary");

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

    fn drain_interface_ready(&mut self, slab_key: usize) -> Result<()> {
        let Some(socket) = self.sockets.get(slab_key) else {
            return Err(eyre!("unknown interface socket token"));
        };
        match &socket.io {
            InterfaceIo::Udp { .. } => self.drain_udp_ready(slab_key),
            #[cfg(any(not(target_os = "linux"), test))]
            InterfaceIo::TcpInterface => Ok(()),
            InterfaceIo::TcpListener { .. } => self.drain_tcp_accept_ready(slab_key),
        }
    }

    fn drain_tun_ready(&mut self) -> Result<()> {
        for _ in 0..TUN_DRAIN_BUDGET {
            if !self.handle_one_tun_packet()? {
                break;
            }
        }
        self.flush_tcp_writes("end of TUN drain slice");
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
        self.counters.tun_rx_sizes.observe(packet_len);

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
        match self.transport_mode {
            TransportMode::Udp => {
                let send_result = send_via_route(&self.sockets, self.udp_port, packet, route);
                if self.handle_udp_send_result(send_result, route, "sending packet from TUN")? {
                    self.counters.tun_to_udp_packets += 1;
                }
            }
            TransportMode::Tcp => {
                let _queued = self.queue_tcp_tun_buffer_via_route(
                    packet_len,
                    route,
                    "sending packet from TUN",
                )?;
            }
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
                self.counters.tun_tx_sizes.observe(sent);
                Ok(true)
            }
            Ok(PacketWriteOutcome::WouldBlock) => {
                self.counters.tun_send_would_block_drops += 1;
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    fn send_overlay_via_route(
        &mut self,
        packet: &[u8],
        route: FastFibEntry,
        context: &str,
    ) -> Result<bool> {
        match self.transport_mode {
            TransportMode::Udp => {
                let send_result = send_via_route(&self.sockets, self.udp_port, packet, route);
                self.handle_udp_send_result(send_result, route, context)
            }
            TransportMode::Tcp => {
                self.queue_tcp_via_route(packet, route, TcpFrameOrigin::Forwarded, context)
            }
        }
    }

    fn queue_tcp_via_route(
        &mut self,
        packet: &[u8],
        route: FastFibEntry,
        origin: TcpFrameOrigin,
        context: &str,
    ) -> Result<bool> {
        let stream_key = match self.ensure_tcp_stream(route, context) {
            Ok(stream_key) => stream_key,
            Err(err) => {
                self.counters.tcp_stream_errors += 1;
                tracing::warn!(
                    socket_slot = route.socket_slot,
                    next_hop = %route.next_hop_ll,
                    context,
                    error = %err,
                    "failed to prepare TCP stream for dataplane packet"
                );
                self.best_effort_reconcile();
                return Ok(false);
            }
        };

        let outcome = {
            let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                return Ok(false);
            };
            stream
                .write_buf
                .queue_frame(packet, origin, TCP_PENDING_LIMIT_BYTES)
        };
        self.finish_tcp_queue_outcome(stream_key, outcome, route, packet.len())
    }

    fn queue_tcp_tun_buffer_via_route(
        &mut self,
        packet_len: usize,
        route: FastFibEntry,
        context: &str,
    ) -> Result<bool> {
        let stream_key = match self.ensure_tcp_stream(route, context) {
            Ok(stream_key) => stream_key,
            Err(err) => {
                self.counters.tcp_stream_errors += 1;
                tracing::warn!(
                    socket_slot = route.socket_slot,
                    next_hop = %route.next_hop_ll,
                    context,
                    error = %err,
                    "failed to prepare TCP stream for dataplane packet"
                );
                self.best_effort_reconcile();
                return Ok(false);
            }
        };

        let outcome = {
            let packet = &self.tun_buf[..packet_len];
            let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                return Ok(false);
            };
            stream.write_buf.queue_frame(
                packet,
                TcpFrameOrigin::TunIngress,
                TCP_PENDING_LIMIT_BYTES,
            )
        };
        self.finish_tcp_queue_outcome(stream_key, outcome, route, packet_len)
    }

    fn finish_tcp_queue_outcome(
        &mut self,
        stream_key: usize,
        outcome: TcpQueueOutcome,
        route: FastFibEntry,
        packet_len: usize,
    ) -> Result<bool> {
        match outcome {
            TcpQueueOutcome::Queued => {
                self.counters.tcp_queued_packets += 1;
                let pending_len = self
                    .tcp_streams
                    .get(stream_key)
                    .map(|stream| stream.write_buf.pending_len())
                    .unwrap_or_default();
                if pending_len >= self.tcp_batch_target_bytes {
                    self.flush_tcp_stream(stream_key, "TCP batch reached target size")?;
                } else {
                    self.reregister_tcp_stream_interest(stream_key)?;
                }
                Ok(true)
            }
            TcpQueueOutcome::Full => {
                self.counters.tcp_queue_drops += 1;
                tracing::debug!(
                    socket_slot = route.socket_slot,
                    next_hop = %route.next_hop_ll,
                    bytes = packet_len,
                    "dropping dataplane packet because TCP stream queue is full"
                );
                Ok(false)
            }
            TcpQueueOutcome::Oversized => {
                self.counters.invalid_packet_drops += 1;
                tracing::warn!(
                    socket_slot = route.socket_slot,
                    next_hop = %route.next_hop_ll,
                    bytes = packet_len,
                    "dropping dataplane packet too large for TCP frame"
                );
                Ok(false)
            }
        }
    }

    fn ensure_tcp_stream(&mut self, route: FastFibEntry, context: &str) -> Result<usize> {
        let peer_key = TcpPeerKey {
            socket_slot: route.socket_slot,
            next_hop_ll: route.next_hop_ll,
        };
        if let Some(stream_key) = self.tcp_peer_to_stream.get(&peer_key).copied()
            && self.tcp_streams.contains(stream_key)
        {
            return Ok(stream_key);
        }
        self.tcp_peer_to_stream.remove(&peer_key);

        let Some(interface) = self.sockets.get(route.socket_slot) else {
            return Err(eyre!("stale TCP socket slot {}", route.socket_slot));
        };
        let peer = SocketAddr::V6(SocketAddrV6::new(
            route.next_hop_ll,
            self.udp_port,
            0,
            interface.ifindex,
        ));
        let ifname = interface.ifname.clone();
        let ifindex = interface.ifindex;

        let (stream, connect_outcome) =
            open_tcp_stream(peer, ifname.as_ref(), ifindex, self.tcp_socket_buffer_bytes)
                .wrap_err_with(|| format!("opening TCP stream via {ifname}"))?;
        let connected = connect_outcome == TcpConnectOutcome::Connected;
        let stream_key = self.insert_registered_tcp_stream(
            route.socket_slot,
            Some(peer_key),
            peer,
            stream,
            connected,
        )?;
        self.tcp_peer_to_stream.insert(peer_key, stream_key);
        self.counters.tcp_connects += 1;
        tracing::debug!(
            interface = %ifname,
            ifindex,
            peer = %peer,
            connected,
            context,
            "dataplane opened TCP stream"
        );
        Ok(stream_key)
    }

    fn insert_registered_tcp_stream(
        &mut self,
        interface_slot: usize,
        peer_key: Option<TcpPeerKey>,
        peer: SocketAddr,
        mut stream: MioTcpStream,
        connected: bool,
    ) -> Result<usize> {
        stream.set_nodelay(true).wrap_err("setting TCP_NODELAY")?;
        let entry = self.tcp_streams.vacant_entry();
        let stream_key = entry.key();
        let interest = if connected {
            Interest::READABLE
        } else {
            Interest::READABLE | Interest::WRITABLE
        };
        let write_interest = !connected;
        self.poll
            .registry()
            .register(&mut stream, tcp_stream_token(stream_key), interest)
            .wrap_err_with(|| format!("registering TCP stream for {peer}"))?;
        entry.insert(TcpPeerStream {
            interface_slot,
            peer_key,
            peer,
            stream,
            connected,
            write_interest,
            read_buf: TcpFrameDecoder::new(TCP_READ_BUFFER_BYTES),
            write_buf: TcpWriteBuffer::new(self.tcp_batch_target_bytes),
        });
        Ok(stream_key)
    }

    fn drain_tcp_accept_ready(&mut self, interface_slot: usize) -> Result<()> {
        for _ in 0..TCP_ACCEPT_DRAIN_BUDGET {
            let accept_result = {
                let Some(interface) = self.sockets.get_mut(interface_slot) else {
                    return Err(eyre!("unknown TCP listener token"));
                };
                let InterfaceIo::TcpListener { listener } = &mut interface.io else {
                    return Err(eyre!("interface socket is not a TCP listener"));
                };
                listener.accept()
            };

            let (stream, peer) = match accept_result {
                Ok(accepted) => accepted,
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(err) => return Err(err).wrap_err("accepting TCP dataplane stream"),
            };

            let Some(validated_slot) =
                self.validated_accepted_tcp_peer_slot(interface_slot, &stream, peer)
            else {
                self.counters.tcp_rejected_peers += 1;
                tracing::warn!(
                    listener_slot = interface_slot,
                    peer = %peer,
                    "rejecting TCP dataplane stream from non-admitted peer"
                );
                continue;
            };

            self.insert_registered_tcp_stream(validated_slot, None, peer, stream, true)?;
            self.counters.tcp_accepts += 1;
            tracing::debug!(
                peer = %peer,
                interface_slot = validated_slot,
                "dataplane accepted TCP stream"
            );
        }
        Ok(())
    }

    fn validated_accepted_tcp_peer_slot(
        &self,
        listener_slot: usize,
        stream: &MioTcpStream,
        peer: SocketAddr,
    ) -> Option<usize> {
        let local = stream.local_addr().ok();

        #[cfg(target_os = "linux")]
        let listener_ifindex = self.sockets.get(listener_slot).map(|socket| socket.ifindex);
        #[cfg(not(target_os = "linux"))]
        let listener_ifindex = {
            let _ = listener_slot;
            None
        };

        admitted_tcp_peer_slot_for_accepted_addrs(
            &self.sockets,
            self.fib.as_ref(),
            peer,
            local,
            listener_ifindex,
        )
    }

    fn drain_tcp_stream_ready(
        &mut self,
        stream_key: usize,
        readable: bool,
        writable: bool,
    ) -> Result<()> {
        if writable {
            self.finish_tcp_connect(stream_key)?;
            self.flush_tcp_stream(stream_key, "TCP stream writable")?;
        }
        if readable {
            self.drain_tcp_stream_read(stream_key)?;
            self.flush_tcp_writes("end of TCP stream drain slice");
        }
        Ok(())
    }

    fn finish_tcp_connect(&mut self, stream_key: usize) -> Result<()> {
        let mut close_error = None;
        if let Some(stream) = self.tcp_streams.get_mut(stream_key)
            && !stream.connected
        {
            match stream.stream.take_error() {
                Ok(None) => {
                    stream.connected = true;
                    tracing::debug!(peer = %stream.peer, "TCP dataplane stream connected");
                }
                Ok(Some(err)) => close_error = Some(err),
                Err(err) => close_error = Some(err),
            }
        }

        if let Some(err) = close_error {
            self.counters.tcp_stream_errors += 1;
            tracing::warn!(error = %err, "TCP dataplane stream connect failed");
            self.close_tcp_stream(stream_key);
        }
        Ok(())
    }

    fn drain_tcp_stream_read(&mut self, stream_key: usize) -> Result<()> {
        for _ in 0..TCP_STREAM_READ_DRAIN_BUDGET {
            let mut decoder = {
                let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                    return Ok(());
                };
                std::mem::take(&mut stream.read_buf)
            };

            let read_result = {
                let Some(stream) = self.tcp_streams.get(stream_key) else {
                    return Ok(());
                };
                decoder.read_from(&stream.stream, TCP_READ_BUFFER_BYTES)
            };
            let read_len = match read_result {
                Ok(0) => {
                    self.close_tcp_stream(stream_key);
                    return Ok(());
                }
                Ok(read_len) => read_len,
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    if let Some(stream) = self.tcp_streams.get_mut(stream_key) {
                        stream.read_buf = decoder;
                    }
                    break;
                }
                Err(err) => {
                    self.counters.tcp_stream_errors += 1;
                    tracing::warn!(error = %err, "failed to read TCP dataplane stream");
                    self.close_tcp_stream(stream_key);
                    return Ok(());
                }
            };
            self.counters.tcp_rx_batches += 1;

            let mut close_for_bad_frame = false;
            loop {
                match decoder.next_frame_range(self.tun_mtu) {
                    Ok(Some(frame_range)) => {
                        let frame_len = frame_range.end.saturating_sub(frame_range.start);
                        self.counters.tcp_rx_frames += 1;
                        self.counters.tcp_rx_bytes += frame_len as u64;
                        self.handle_tcp_frame(&mut decoder.bytes[frame_range])?;
                        decoder.finish_frame(TCP_READ_BUFFER_BYTES);
                    }
                    Ok(None) => break,
                    Err(err) => {
                        self.counters.tcp_frame_errors += 1;
                        tracing::warn!(?err, "invalid TCP dataplane frame");
                        close_for_bad_frame = true;
                        break;
                    }
                }
            }

            if close_for_bad_frame {
                self.close_tcp_stream(stream_key);
                return Ok(());
            }
            let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                return Ok(());
            };
            stream.read_buf = decoder;

            if read_len < TCP_READ_BUFFER_BYTES {
                break;
            }
        }
        Ok(())
    }

    fn handle_tcp_frame(&mut self, packet: &mut [u8]) -> Result<()> {
        let packet_len = packet.len();
        let Some(dst) = ipv6_destination(packet) else {
            self.counters.invalid_packet_drops += 1;
            tracing::debug!("dropping invalid TCP payload");
            return Ok(());
        };

        if self.is_local(dst) {
            let write_result = write_tun_packet(&self.tun_device, packet);
            if self.handle_tun_write_result(write_result)? {
                self.counters.local_delivered_packets += 1;
            }
            return Ok(());
        }

        if !has_forwardable_hop_limit(packet) {
            self.counters.hop_limit_drops += 1;
            tracing::debug!(destination = %dst, "dropping TCP packet with exhausted hop limit");
            return Ok(());
        }

        let Some(route) = self.lookup_fast_route(dst) else {
            self.counters.no_route_drops += 1;
            tracing::debug!(destination = %dst, "dataplane has no route for forwarded TCP packet");
            return Ok(());
        };

        debug_assert!(decrement_hop_limit(packet));
        tracing::debug!(
            destination = %dst,
            socket_slot = route.socket_slot,
            next_hop = %route.next_hop_ll,
            bytes = packet_len,
            "dataplane resolved route for forwarded TCP packet"
        );
        let _queued = self.send_overlay_via_route(packet, route, "forwarding TCP packet")?;
        Ok(())
    }

    fn flush_tcp_writes(&mut self, context: &str) {
        if self.transport_mode != TransportMode::Tcp {
            return;
        }
        let keys: Vec<usize> = self
            .tcp_streams
            .iter()
            .filter_map(|(key, stream)| {
                (stream.connected && stream.write_buf.has_pending()).then_some(key)
            })
            .collect();
        for key in keys {
            if let Err(err) = self.flush_tcp_stream(key, context) {
                self.counters.tcp_stream_errors += 1;
                tracing::warn!(error = %err, context, "failed to flush TCP dataplane stream");
                self.close_tcp_stream(key);
            }
        }
    }

    fn flush_tcp_stream(&mut self, stream_key: usize, context: &str) -> Result<()> {
        let mut wrote_bytes = 0u64;
        let mut completed_frames = TcpCompletedFrames::default();
        let mut write_calls = 0u64;
        let mut partial_writes = 0u64;
        let mut blocked_writes = 0u64;
        let mut close_error = None;

        {
            let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                return Ok(());
            };
            if !stream.connected {
                return Ok(());
            }
            while stream.write_buf.has_pending() {
                let pending_len = stream.write_buf.pending_len();
                match stream.stream.write(stream.write_buf.pending()) {
                    Ok(0) => {
                        blocked_writes += 1;
                        break;
                    }
                    Ok(written) => {
                        wrote_bytes += written as u64;
                        write_calls += 1;
                        if written < pending_len {
                            partial_writes += 1;
                        }
                        let completed = stream
                            .write_buf
                            .consume(written, self.tcp_batch_target_bytes);
                        completed_frames.total += completed.total;
                        completed_frames.tun_ingress += completed.tun_ingress;
                        completed_frames.forwarded += completed.forwarded;
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        blocked_writes += 1;
                        break;
                    }
                    Err(err) => {
                        close_error = Some(err);
                        break;
                    }
                }
            }
        }

        self.counters.tcp_tx_bytes += wrote_bytes;
        self.counters.tcp_written_frames += completed_frames.total as u64;
        self.counters.tun_to_udp_packets += completed_frames.tun_ingress as u64;
        self.counters.udp_forwarded_packets += completed_frames.forwarded as u64;
        self.counters.tcp_tx_batches += write_calls;
        self.counters.tcp_partial_writes += partial_writes;
        self.counters.tcp_blocked_writes += blocked_writes;

        if let Some(err) = close_error {
            return Err(err).wrap_err("writing TCP dataplane stream");
        }

        if wrote_bytes != 0 {
            tracing::debug!(
                bytes = wrote_bytes,
                write_calls,
                context,
                "flushed TCP dataplane stream"
            );
        }
        self.reregister_tcp_stream_interest(stream_key)?;
        Ok(())
    }

    fn reregister_tcp_stream_interest(&mut self, stream_key: usize) -> Result<()> {
        {
            let Some(stream) = self.tcp_streams.get_mut(stream_key) else {
                return Ok(());
            };
            let needs_write = !stream.connected || stream.write_buf.has_pending();
            if stream.write_interest == needs_write {
                return Ok(());
            }
            let interest = if needs_write {
                Interest::READABLE | Interest::WRITABLE
            } else {
                Interest::READABLE
            };
            self.poll
                .registry()
                .reregister(&mut stream.stream, tcp_stream_token(stream_key), interest)
                .wrap_err_with(|| format!("reregistering TCP stream for {}", stream.peer))?;
            stream.write_interest = needs_write;
        }
        self.counters.tcp_reregisters += 1;
        Ok(())
    }

    fn close_tcp_stream(&mut self, stream_key: usize) {
        let Some(mut stream) = self.tcp_streams.try_remove(stream_key) else {
            return;
        };
        if let Some(peer_key) = stream.peer_key {
            self.tcp_peer_to_stream.remove(&peer_key);
        }
        if let Err(err) = self.poll.registry().deregister(&mut stream.stream) {
            tracing::warn!(
                peer = %stream.peer,
                error = %err,
                "failed to deregister TCP dataplane stream"
            );
        }
    }

    fn remove_tcp_streams_for_interface(&mut self, interface_slot: usize) {
        let keys: Vec<usize> = self
            .tcp_streams
            .iter()
            .filter_map(|(key, stream)| (stream.interface_slot == interface_slot).then_some(key))
            .collect();
        for key in keys {
            self.close_tcp_stream(key);
        }
    }

    fn apply_snapshot(&mut self, snapshot: Arc<FibSnapshot>) -> Result<()> {
        self.flush_tcp_writes("dataplane snapshot update");
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
        self.ensure_tcp_listener_for_admitted_interfaces()?;
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
        let slab_key = match self.transport_mode {
            TransportMode::Udp => {
                let socket = open_udp_socket(self.udp_port, ifname.as_ref(), ifindex)
                    .wrap_err_with(|| format!("opening UDP socket for {}", ifname))?;
                self.insert_registered_udp_socket(ifname.clone(), ifindex, socket)?
            }
            TransportMode::Tcp => self.add_tcp_socket(ifname.clone(), ifindex)?,
        };
        self.ifname_to_slab.insert(ifname, slab_key);
        Ok(())
    }

    fn add_tcp_socket(&mut self, ifname: Box<str>, ifindex: u32) -> Result<usize> {
        #[cfg(target_os = "linux")]
        {
            let listener = open_tcp_listener(
                self.udp_port,
                ifname.as_ref(),
                ifindex,
                self.tcp_socket_buffer_bytes,
            )
            .wrap_err_with(|| format!("opening TCP listener for {}", ifname))?;
            self.insert_registered_tcp_listener(ifname, ifindex, listener)
        }

        #[cfg(not(target_os = "linux"))]
        {
            if self.has_tcp_listener() {
                Ok(self.insert_registered_tcp_interface(ifname, ifindex))
            } else {
                let listener = open_tcp_listener(
                    self.udp_port,
                    ifname.as_ref(),
                    ifindex,
                    self.tcp_socket_buffer_bytes,
                )
                .wrap_err_with(|| format!("opening TCP listener for {}", ifname))?;
                self.insert_registered_tcp_listener(ifname, ifindex, listener)
            }
        }
    }

    fn refresh_socket(&mut self, ifname: &str, ifindex: u32) -> Result<()> {
        if self.transport_mode == TransportMode::Tcp {
            self.remove_socket(ifname)?;
            return self.add_socket(ifname.into(), ifindex);
        }

        let Some(old_slab_key) = self.ifname_to_slab.get(ifname).copied() else {
            return self.add_socket(ifname.into(), ifindex);
        };

        let new_ifname: Box<str> = ifname.into();
        let new_socket = open_udp_socket(self.udp_port, ifname, ifindex)
            .wrap_err_with(|| format!("opening replacement UDP socket for {}", ifname))?;
        let new_slab_key =
            self.insert_registered_udp_socket(new_ifname.clone(), ifindex, new_socket)?;
        let previous = self.ifname_to_slab.insert(new_ifname.clone(), new_slab_key);
        debug_assert_eq!(previous, Some(old_slab_key));

        let Some(mut old_socket) = self.sockets.try_remove(old_slab_key) else {
            return Ok(());
        };
        if let InterfaceIo::Udp { socket, .. } = &mut old_socket.io
            && let Err(err) = self.poll.registry().deregister(socket)
        {
            tracing::warn!(
                interface = %ifname,
                error = %err,
                "failed to deregister old dataplane socket after refresh"
            );
        }
        Ok(())
    }

    fn insert_registered_udp_socket(
        &mut self,
        ifname: Box<str>,
        ifindex: u32,
        mut socket: MioUdpSocket,
    ) -> Result<usize> {
        let udp_state = UdpSocketState::new(UdpSockRef::from(&socket))
            .wrap_err_with(|| format!("initializing UDP fast-path state for {}", ifname))?;
        let entry = self.sockets.vacant_entry();
        let slab_key = entry.key();
        self.poll
            .registry()
            .register(&mut socket, interface_token(slab_key), Interest::READABLE)
            .wrap_err_with(|| format!("registering UDP socket for {}", ifname))?;
        entry.insert(InterfaceSocket {
            ifname,
            ifindex,
            io: InterfaceIo::Udp { socket, udp_state },
        });
        Ok(slab_key)
    }

    fn insert_registered_tcp_listener(
        &mut self,
        ifname: Box<str>,
        ifindex: u32,
        mut listener: MioTcpListener,
    ) -> Result<usize> {
        let entry = self.sockets.vacant_entry();
        let slab_key = entry.key();
        self.poll
            .registry()
            .register(&mut listener, interface_token(slab_key), Interest::READABLE)
            .wrap_err_with(|| format!("registering TCP listener for {}", ifname))?;
        entry.insert(InterfaceSocket {
            ifname,
            ifindex,
            io: InterfaceIo::TcpListener { listener },
        });
        Ok(slab_key)
    }

    #[cfg(not(target_os = "linux"))]
    fn insert_registered_tcp_interface(&mut self, ifname: Box<str>, ifindex: u32) -> usize {
        let entry = self.sockets.vacant_entry();
        let slab_key = entry.key();
        entry.insert(InterfaceSocket {
            ifname,
            ifindex,
            io: InterfaceIo::TcpInterface,
        });
        slab_key
    }

    #[cfg(not(target_os = "linux"))]
    fn has_tcp_listener(&self) -> bool {
        self.sockets
            .iter()
            .any(|(_, socket)| matches!(socket.io, InterfaceIo::TcpListener { .. }))
    }

    fn ensure_tcp_listener_for_admitted_interfaces(&mut self) -> Result<()> {
        if self.transport_mode != TransportMode::Tcp {
            return Ok(());
        }

        #[cfg(target_os = "linux")]
        {
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            if self.fib.admitted_interfaces.is_empty() || self.has_tcp_listener() {
                return Ok(());
            }

            let Some((ifname, slot)) = self
                .ifname_to_slab
                .iter()
                .min_by(|(left, _), (right, _)| left.cmp(right))
                .map(|(ifname, slot)| (ifname.clone(), *slot))
            else {
                return Ok(());
            };
            let Some(socket) = self.sockets.get_mut(slot) else {
                return Ok(());
            };
            let listener = open_tcp_listener(
                self.udp_port,
                ifname.as_ref(),
                socket.ifindex,
                self.tcp_socket_buffer_bytes,
            )
            .wrap_err_with(|| format!("opening TCP listener for {}", ifname))?;
            let mut listener = listener;
            self.poll
                .registry()
                .register(&mut listener, interface_token(slot), Interest::READABLE)
                .wrap_err_with(|| format!("registering TCP listener for {}", ifname))?;
            socket.io = InterfaceIo::TcpListener { listener };
            tracing::info!(
                interface = %ifname,
                ifindex = socket.ifindex,
                "promoted admitted TCP interface to wildcard listener"
            );
            Ok(())
        }
    }

    fn remove_socket(&mut self, ifname: &str) -> Result<()> {
        let Some(slab_key) = self.ifname_to_slab.remove(ifname) else {
            return Ok(());
        };
        self.remove_tcp_streams_for_interface(slab_key);
        let Some(mut socket) = self.sockets.try_remove(slab_key) else {
            return Ok(());
        };
        match &mut socket.io {
            InterfaceIo::Udp { socket, .. } => self
                .poll
                .registry()
                .deregister(socket)
                .wrap_err_with(|| format!("deregistering UDP socket for {}", ifname))?,
            #[cfg(any(not(target_os = "linux"), test))]
            InterfaceIo::TcpInterface => {}
            InterfaceIo::TcpListener { listener } => self
                .poll
                .registry()
                .deregister(listener)
                .wrap_err_with(|| format!("deregistering TCP listener for {}", ifname))?,
        }
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
            tun_rx_size_le_16k_delta = delta.tun_rx_sizes.le_16k,
            tun_rx_size_le_32k_delta = delta.tun_rx_sizes.le_32k,
            tun_rx_size_le_48k_delta = delta.tun_rx_sizes.le_48k,
            tun_rx_size_le_60k_delta = delta.tun_rx_sizes.le_60k,
            tun_rx_size_le_65k_delta = delta.tun_rx_sizes.le_65k,
            tun_rx_size_gt_65k_delta = delta.tun_rx_sizes.gt_65k,
            udp_rx_packets_delta = delta.udp_rx_packets,
            udp_rx_bytes_delta = delta.udp_rx_bytes,
            udp_tx_packets_delta = delta.udp_tx_packets,
            udp_tx_bytes_delta = delta.udp_tx_bytes,
            tun_tx_packets_delta = delta.tun_tx_packets,
            tun_tx_bytes_delta = delta.tun_tx_bytes,
            tun_tx_size_le_16k_delta = delta.tun_tx_sizes.le_16k,
            tun_tx_size_le_32k_delta = delta.tun_tx_sizes.le_32k,
            tun_tx_size_le_48k_delta = delta.tun_tx_sizes.le_48k,
            tun_tx_size_le_60k_delta = delta.tun_tx_sizes.le_60k,
            tun_tx_size_le_65k_delta = delta.tun_tx_sizes.le_65k,
            tun_tx_size_gt_65k_delta = delta.tun_tx_sizes.gt_65k,
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
            tcp_connects_delta = delta.tcp_connects,
            tcp_accepts_delta = delta.tcp_accepts,
            tcp_rejected_peers_delta = delta.tcp_rejected_peers,
            tcp_reconnects_delta = delta.tcp_reconnects,
            tcp_queued_packets_delta = delta.tcp_queued_packets,
            tcp_tx_batches_delta = delta.tcp_tx_batches,
            tcp_written_frames_delta = delta.tcp_written_frames,
            tcp_tx_bytes_delta = delta.tcp_tx_bytes,
            tcp_reregisters_delta = delta.tcp_reregisters,
            tcp_rx_batches_delta = delta.tcp_rx_batches,
            tcp_rx_frames_delta = delta.tcp_rx_frames,
            tcp_rx_bytes_delta = delta.tcp_rx_bytes,
            tcp_partial_writes_delta = delta.tcp_partial_writes,
            tcp_blocked_writes_delta = delta.tcp_blocked_writes,
            tcp_queue_drops_delta = delta.tcp_queue_drops,
            tcp_frame_errors_delta = delta.tcp_frame_errors,
            tcp_stream_errors_delta = delta.tcp_stream_errors,
            tun_rx_packets_total = self.counters.tun_rx_packets,
            tun_rx_size_le_16k_total = self.counters.tun_rx_sizes.le_16k,
            tun_rx_size_le_32k_total = self.counters.tun_rx_sizes.le_32k,
            tun_rx_size_le_48k_total = self.counters.tun_rx_sizes.le_48k,
            tun_rx_size_le_60k_total = self.counters.tun_rx_sizes.le_60k,
            tun_rx_size_le_65k_total = self.counters.tun_rx_sizes.le_65k,
            tun_rx_size_gt_65k_total = self.counters.tun_rx_sizes.gt_65k,
            udp_rx_packets_total = self.counters.udp_rx_packets,
            udp_tx_packets_total = self.counters.udp_tx_packets,
            tun_tx_packets_total = self.counters.tun_tx_packets,
            tun_tx_size_le_16k_total = self.counters.tun_tx_sizes.le_16k,
            tun_tx_size_le_32k_total = self.counters.tun_tx_sizes.le_32k,
            tun_tx_size_le_48k_total = self.counters.tun_tx_sizes.le_48k,
            tun_tx_size_le_60k_total = self.counters.tun_tx_sizes.le_60k,
            tun_tx_size_le_65k_total = self.counters.tun_tx_sizes.le_65k,
            tun_tx_size_gt_65k_total = self.counters.tun_tx_sizes.gt_65k,
            no_route_drops_total = self.counters.no_route_drops,
            udp_send_would_block_drops_total = self.counters.udp_send_would_block_drops,
            tun_send_would_block_drops_total = self.counters.tun_send_would_block_drops,
            udp_send_errors_total = self.counters.udp_send_errors,
            tcp_rejected_peers_total = self.counters.tcp_rejected_peers,
            tcp_queued_packets_total = self.counters.tcp_queued_packets,
            tcp_written_frames_total = self.counters.tcp_written_frames,
            tcp_reregisters_total = self.counters.tcp_reregisters,
            tcp_rx_batches_total = self.counters.tcp_rx_batches,
            tcp_rx_frames_total = self.counters.tcp_rx_frames,
            tcp_queue_drops_total = self.counters.tcp_queue_drops,
            tcp_frame_errors_total = self.counters.tcp_frame_errors,
            tcp_stream_errors_total = self.counters.tcp_stream_errors,
            "dataplane counters"
        );
    }
}

fn recv_udp_packets(
    socket: &InterfaceSocket,
    batch: &mut UdpRecvBatch,
    max_packets: usize,
) -> Result<UdpRecvOutcome> {
    let InterfaceIo::Udp {
        socket: udp_socket,
        udp_state,
    } = &socket.io
    else {
        return Err(eyre!("interface {} is not a UDP socket", socket.ifname));
    };
    match batch.recv_from(udp_socket, udp_state, max_packets) {
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
    let InterfaceIo::Udp {
        socket: udp_socket,
        udp_state,
    } = &socket.io
    else {
        return Err(eyre!("route socket slot {} is not UDP", route.socket_slot));
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
    match udp_socket.try_io(|| udp_state.try_send(UdpSockRef::from(udp_socket), &transmit)) {
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

fn admitted_tcp_peer_slot_by_ifindex(
    sockets: &Slab<InterfaceSocket>,
    fib: &FibSnapshot,
    peer_addr: Ipv6Addr,
    ifindex: u32,
) -> Option<usize> {
    if ifindex == 0 || !peer_addr.is_unicast_link_local() {
        return None;
    }

    sockets.iter().find_map(|(slot, socket)| {
        (socket.ifindex == ifindex
            && fib.admitted_interfaces.contains(socket.ifname.as_ref())
            && fib.admitted_neighbours.iter().any(|neighbour| {
                neighbour.ifname.as_ref() == socket.ifname.as_ref()
                    && neighbour.link_local == peer_addr
            }))
        .then_some(slot)
    })
}

fn admitted_tcp_peer_slot_for_accepted_addrs(
    sockets: &Slab<InterfaceSocket>,
    fib: &FibSnapshot,
    peer: SocketAddr,
    local: Option<SocketAddr>,
    listener_ifindex: Option<u32>,
) -> Option<usize> {
    let SocketAddr::V6(peer_v6) = peer else {
        return None;
    };
    let peer_addr = *peer_v6.ip();
    if !peer_addr.is_unicast_link_local() {
        return None;
    }

    if let Some(slot) =
        admitted_tcp_peer_slot_by_ifindex(sockets, fib, peer_addr, peer_v6.scope_id())
    {
        return Some(slot);
    }

    if let Some(SocketAddr::V6(local_v6)) = local {
        if let Some(slot) =
            admitted_tcp_peer_slot_by_ifindex(sockets, fib, peer_addr, local_v6.scope_id())
        {
            return Some(slot);
        }
        if let Some(slot) =
            admitted_tcp_peer_slot_by_local_addr(sockets, fib, peer_addr, *local_v6.ip())
        {
            return Some(slot);
        }
    }

    listener_ifindex
        .and_then(|ifindex| admitted_tcp_peer_slot_by_ifindex(sockets, fib, peer_addr, ifindex))
}

fn admitted_tcp_peer_slot_by_local_addr(
    sockets: &Slab<InterfaceSocket>,
    fib: &FibSnapshot,
    peer_addr: Ipv6Addr,
    local_addr: Ipv6Addr,
) -> Option<usize> {
    if !peer_addr.is_unicast_link_local() || !local_addr.is_unicast_link_local() {
        return None;
    }

    sockets.iter().find_map(|(slot, socket)| {
        (fib.interface_link_locals
            .get(socket.ifname.as_ref())
            .is_some_and(|addr| *addr == local_addr)
            && fib.admitted_interfaces.contains(socket.ifname.as_ref())
            && fib.admitted_neighbours.iter().any(|neighbour| {
                neighbour.ifname.as_ref() == socket.ifname.as_ref()
                    && neighbour.link_local == peer_addr
            }))
        .then_some(slot)
    })
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

fn open_tcp_listener(
    port: u16,
    ifname: &str,
    ifindex: u32,
    socket_buffer_bytes: usize,
) -> io::Result<MioTcpListener> {
    let socket = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_only_v6(true)?;
    socket.set_nonblocking(true)?;
    socket.set_recv_buffer_size(socket_buffer_bytes)?;
    socket.set_send_buffer_size(socket_buffer_bytes)?;

    #[cfg(not(target_os = "linux"))]
    let _ = (ifname, ifindex);

    #[cfg(target_os = "linux")]
    {
        let Some(ifindex) = NonZeroU32::new(ifindex) else {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!("invalid ifindex for {ifname}"),
            ));
        };
        socket.bind_device(Some(ifname.as_bytes()))?;
        socket.bind_device_by_index_v6(Some(ifindex))?;
    }
    socket.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, port, 0, 0).into())?;
    socket.listen(1024)?;

    let listener: TcpListener = socket.into();
    Ok(MioTcpListener::from_std(listener))
}

fn open_tcp_stream(
    peer: SocketAddr,
    ifname: &str,
    ifindex: u32,
    socket_buffer_bytes: usize,
) -> io::Result<(MioTcpStream, TcpConnectOutcome)> {
    let socket = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_only_v6(true)?;
    socket.set_nonblocking(true)?;
    socket.set_tcp_nodelay(true)?;
    socket.set_recv_buffer_size(socket_buffer_bytes)?;
    socket.set_send_buffer_size(socket_buffer_bytes)?;

    let Some(ifindex) = NonZeroU32::new(ifindex) else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("invalid ifindex for {ifname}"),
        ));
    };

    #[cfg(target_os = "linux")]
    socket.bind_device(Some(ifname.as_bytes()))?;
    socket.bind_device_by_index_v6(Some(ifindex))?;

    let outcome = match socket.connect(&peer.into()) {
        Ok(()) => TcpConnectOutcome::Connected,
        Err(err) if is_tcp_connect_in_progress(&err) => TcpConnectOutcome::InProgress,
        Err(err) => return Err(err),
    };

    let stream: TcpStream = socket.into();
    Ok((MioTcpStream::from_std(stream), outcome))
}

fn is_tcp_connect_in_progress(err: &io::Error) -> bool {
    matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted)
        || matches!(
            err.raw_os_error(),
            Some(code)
                if code == libc::EINPROGRESS
                    || code == libc::EALREADY
                    || code == libc::EWOULDBLOCK
        )
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
    use std::io::ErrorKind;
    use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket as StdUdpSocket};
    use std::time::Duration;

    use ahash::RandomState;
    use hashbrown::{HashMap, HashSet};
    use mio::net::UdpSocket as MioUdpSocket;
    use slab::Slab;

    use crate::config::TCP_PENDING_LIMIT_BYTES;
    use crate::fib::{AdmittedNeighbour, FibEntry, FibSnapshot, host_key};

    use super::{
        BATCH_SIZE, FastFibEntry, InterfaceIo, InterfaceSocket, RecvMeta, SocketAction,
        TCP_READ_BUFFER_BYTES, TcpFrameDecoder, TcpFrameError, TcpFrameOrigin, TcpQueueOutcome,
        TcpWriteBuffer, UdpRecvBatch, UdpRecvOutcome, UdpSockRef, UdpSocketState,
        admitted_tcp_peer_slot_by_ifindex, admitted_tcp_peer_slot_by_local_addr,
        admitted_tcp_peer_slot_for_accepted_addrs, compile_fast_routes, decrement_hop_limit,
        has_missing_admitted_sockets, ipv6_destination, plan_socket_actions, recv_udp_packets,
        udp_batch_packet_bytes, udp_meta_stride,
    };

    fn sample_ipv6_packet(dst: Ipv6Addr, hop_limit: u8) -> Vec<u8> {
        let mut packet = vec![0u8; 40];
        packet[0] = 0x60;
        packet[6] = 17;
        packet[7] = hop_limit;
        packet[24..40].copy_from_slice(&dst.octets());
        packet
    }

    fn bind_loopback_udp() -> StdUdpSocket {
        for attempt in 0..10 {
            match StdUdpSocket::bind((Ipv6Addr::LOCALHOST, 0)) {
                Ok(socket) => return socket,
                Err(err) if err.kind() == ErrorKind::PermissionDenied && attempt < 9 => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(err) => panic!("binding IPv6 loopback UDP socket failed: {err}"),
            }
        }
        unreachable!("retry loop should return or panic")
    }

    fn next_decoded_frame(decoder: &mut TcpFrameDecoder, max_packet_len: usize) -> Option<Vec<u8>> {
        let range = decoder.next_frame_range(max_packet_len).unwrap()?;
        let frame = decoder.bytes[range].to_vec();
        decoder.finish_frame(TCP_READ_BUFFER_BYTES);
        Some(frame)
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
    fn tcp_frame_decoder_handles_fragmented_and_coalesced_frames() {
        let packet_a = sample_ipv6_packet("fde0::1".parse().unwrap(), 32);
        let packet_b = sample_ipv6_packet("fde0::2".parse().unwrap(), 32);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(packet_a.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&packet_a);
        bytes.extend_from_slice(&(packet_b.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&packet_b);

        let mut decoder = TcpFrameDecoder::new(TCP_READ_BUFFER_BYTES);
        decoder.push(&bytes[..7]);
        assert_eq!(decoder.next_frame_range(1500).unwrap(), None);

        decoder.push(&bytes[7..]);
        assert_eq!(next_decoded_frame(&mut decoder, 1500), Some(packet_a));
        assert_eq!(next_decoded_frame(&mut decoder, 1500), Some(packet_b));
        assert_eq!(decoder.next_frame_range(1500).unwrap(), None);
    }

    #[test]
    fn tcp_frame_decoder_rejects_invalid_lengths() {
        let mut decoder = TcpFrameDecoder::new(TCP_READ_BUFFER_BYTES);
        decoder.push(&0u16.to_be_bytes());
        assert_eq!(
            decoder.next_frame_range(1500).unwrap_err(),
            TcpFrameError::ZeroLength
        );

        let mut decoder = TcpFrameDecoder::new(TCP_READ_BUFFER_BYTES);
        decoder.push(&2000u16.to_be_bytes());
        assert_eq!(
            decoder.next_frame_range(1500).unwrap_err(),
            TcpFrameError::Oversized {
                len: 2000,
                max: 1500,
            }
        );
    }

    #[test]
    fn tcp_write_buffer_frames_packets_and_tracks_pending_bytes() {
        let packet = sample_ipv6_packet("fde0::1".parse().unwrap(), 32);
        let mut buffer = TcpWriteBuffer::new(256 * 1024);

        assert_eq!(
            buffer.queue_frame(&packet, TcpFrameOrigin::TunIngress, TCP_PENDING_LIMIT_BYTES),
            TcpQueueOutcome::Queued
        );
        assert_eq!(buffer.frame_ends.len(), 1);
        assert_eq!(buffer.pending_len(), packet.len() + 2);
        assert_eq!(&buffer.pending()[..2], &(packet.len() as u16).to_be_bytes());
        assert_eq!(&buffer.pending()[2..], packet.as_slice());

        let completed = buffer.consume(2, 256 * 1024);
        assert_eq!(completed.total, 0);
        assert_eq!(buffer.pending_len(), packet.len());
        let completed = buffer.consume(packet.len(), 256 * 1024);
        assert_eq!(completed.total, 1);
        assert_eq!(completed.tun_ingress, 1);
        assert_eq!(completed.forwarded, 0);
        assert!(!buffer.has_pending());
        assert_eq!(buffer.frame_ends.len(), 0);
    }

    #[test]
    fn udp_batch_recv_returns_single_datagram_without_full_batch() {
        let receiver = bind_loopback_udp();
        let receiver_addr = receiver.local_addr().unwrap();
        let receiver = MioUdpSocket::from_std(receiver);
        let udp_state = UdpSocketState::new(UdpSockRef::from(&receiver)).unwrap();
        let socket = InterfaceSocket {
            ifname: "lo0".into(),
            ifindex: 0,
            io: InterfaceIo::Udp {
                socket: receiver,
                udp_state,
            },
        };

        let sender = bind_loopback_udp();
        let payload = sample_ipv6_packet("fde0::1234".parse().unwrap(), 32);
        sender.send_to(&payload, receiver_addr).unwrap();

        let mut batch = UdpRecvBatch::new(payload.len());
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
    fn tcp_mode_udp_batch_keeps_udp_sized_buffers() {
        assert_eq!(
            udp_batch_packet_bytes(crate::config::TransportMode::Udp, 9000),
            9000
        );
        assert_eq!(
            udp_batch_packet_bytes(crate::config::TransportMode::Tcp, 9000),
            usize::from(crate::config::UDP_TUN_MTU)
        );
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
            admitted_neighbours: HashSet::with_hasher(RandomState::new()),
            interface_link_locals: HashMap::with_hasher(RandomState::new()),
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

    #[test]
    fn tcp_peer_admission_requires_link_local_neighbour_on_matching_ifindex() {
        let mut sockets = Slab::new();
        let socket = InterfaceSocket {
            ifname: "en2".into(),
            ifindex: 7,
            io: InterfaceIo::TcpInterface,
        };
        let slot = sockets.insert(socket);

        let peer = "fe80::1".parse::<Ipv6Addr>().unwrap();
        let local = "fe80::2".parse::<Ipv6Addr>().unwrap();
        let fib = FibSnapshot {
            locals: HashSet::with_hasher(RandomState::new()),
            admitted_interfaces: ["en2".into()].into_iter().collect(),
            admitted_neighbours: [AdmittedNeighbour {
                ifname: "en2".into(),
                link_local: peer,
            }]
            .into_iter()
            .collect(),
            interface_link_locals: [("en2".into(), local)].into_iter().collect(),
            routes: HashMap::with_hasher(RandomState::new()),
        };

        assert_eq!(
            admitted_tcp_peer_slot_by_ifindex(&sockets, &fib, peer, 7),
            Some(slot)
        );
        assert_eq!(
            admitted_tcp_peer_slot_by_ifindex(&sockets, &fib, peer, 8),
            None
        );
        assert_eq!(
            admitted_tcp_peer_slot_by_ifindex(&sockets, &fib, "2001:db8::1".parse().unwrap(), 7),
            None
        );
        assert_eq!(
            admitted_tcp_peer_slot_by_local_addr(&sockets, &fib, peer, local),
            Some(slot)
        );
        assert_eq!(
            admitted_tcp_peer_slot_by_local_addr(&sockets, &fib, peer, "fe80::3".parse().unwrap()),
            None
        );
    }

    #[test]
    fn accepted_tcp_peer_admission_uses_scope_and_local_addr_fallbacks() {
        let mut sockets = Slab::new();
        let en2_slot = sockets.insert(InterfaceSocket {
            ifname: "en2".into(),
            ifindex: 7,
            io: InterfaceIo::TcpInterface,
        });
        let en3_slot = sockets.insert(InterfaceSocket {
            ifname: "en3".into(),
            ifindex: 8,
            io: InterfaceIo::TcpInterface,
        });

        let en2_peer = "fe80::1".parse::<Ipv6Addr>().unwrap();
        let en3_peer = "fe80::3".parse::<Ipv6Addr>().unwrap();
        let en2_local = "fe80::2".parse::<Ipv6Addr>().unwrap();
        let en3_local = "fe80::4".parse::<Ipv6Addr>().unwrap();
        let fib = FibSnapshot {
            locals: HashSet::with_hasher(RandomState::new()),
            admitted_interfaces: ["en2".into(), "en3".into()].into_iter().collect(),
            admitted_neighbours: [
                AdmittedNeighbour {
                    ifname: "en2".into(),
                    link_local: en2_peer,
                },
                AdmittedNeighbour {
                    ifname: "en3".into(),
                    link_local: en3_peer,
                },
            ]
            .into_iter()
            .collect(),
            interface_link_locals: [("en2".into(), en2_local), ("en3".into(), en3_local)]
                .into_iter()
                .collect(),
            routes: HashMap::with_hasher(RandomState::new()),
        };

        let scoped_peer = SocketAddr::V6(SocketAddrV6::new(en2_peer, 5201, 0, 7));
        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(&sockets, &fib, scoped_peer, None, None),
            Some(en2_slot)
        );

        let unscoped_peer = SocketAddr::V6(SocketAddrV6::new(en2_peer, 5201, 0, 0));
        let scoped_local = SocketAddr::V6(SocketAddrV6::new(en2_local, 5201, 0, 7));
        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(
                &sockets,
                &fib,
                unscoped_peer,
                Some(scoped_local),
                None,
            ),
            Some(en2_slot)
        );

        let unscoped_local = SocketAddr::V6(SocketAddrV6::new(en3_local, 5201, 0, 0));
        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(
                &sockets,
                &fib,
                SocketAddr::V6(SocketAddrV6::new(en3_peer, 5201, 0, 0)),
                Some(unscoped_local),
                None,
            ),
            Some(en3_slot)
        );

        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(&sockets, &fib, unscoped_peer, None, Some(7),),
            Some(en2_slot)
        );
        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(
                &sockets,
                &fib,
                SocketAddr::V6(SocketAddrV6::new(
                    "2001:db8::1".parse().unwrap(),
                    5201,
                    0,
                    7
                )),
                Some(scoped_local),
                Some(7),
            ),
            None
        );

        let fib_without_en3_admitted = FibSnapshot {
            admitted_interfaces: ["en2".into()].into_iter().collect(),
            ..fib.clone()
        };
        assert_eq!(
            admitted_tcp_peer_slot_for_accepted_addrs(
                &sockets,
                &fib_without_en3_admitted,
                SocketAddr::V6(SocketAddrV6::new(en3_peer, 5201, 0, 0)),
                Some(unscoped_local),
                None,
            ),
            None
        );
    }
}
