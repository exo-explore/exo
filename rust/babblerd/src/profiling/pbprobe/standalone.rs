use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self, ErrorKind};
use std::net::{Ipv6Addr, SocketAddr, UdpSocket};
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{Result, WrapErr, eyre};
use iroh_quinn_udp::{BATCH_SIZE, Transmit, UdpSockRef, UdpSocketState};

use crate::config::{OUTER_IPV6_HEADER_BYTES, OUTER_UDP_HEADER_BYTES};

use super::estimator::{
    AcceptedSample, DEFAULT_CONTROL_RETRIES, DEFAULT_DISPERSION_THRESHOLD_MS, DEFAULT_MAX_BULK_LEN,
    DEFAULT_PBPROBE_PORT, DEFAULT_RTS_TIMEOUT_MS, DEFAULT_SAMPLE_COUNT, DEFAULT_START_TIMEOUT_MS,
    DEFAULT_UTILIZATION, Estimate, EstimateOutcome, PbProbeConfig, next_bulk_len, pacing_interval,
    select_capacity_sample,
};
use super::protocol::{
    HEADER_LEN, Header, PacketKind, ProtocolError, decode_header, decode_header_with_aux,
    duration_nanos, encode_header, encode_header_with_aux,
};
use crate::profiling::socket::{open_link_local_udp, parse_link_local_addr, scoped_peer_addr};

const MAX_UDP_PACKET_BYTES: usize = 65_535;
const SERVER_RECV_TIMEOUT: Duration = Duration::from_secs(1);
const MIN_ATTEMPT_MULTIPLIER: u32 = 3;

pub fn run_from_env() -> Result<()> {
    run_cli(Cli::parse())
}

pub fn run<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString> + Clone,
{
    run_cli(Cli::try_parse_from(args)?)
}

fn run_cli(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Server(options) => run_server(options),
        Command::Client(options) => run_client(options.into_client_options()),
    }
}

#[derive(Debug, Parser)]
#[command(name = "pbprobe_link")]
#[command(about = "Run standalone PBProbe link-local capacity probes")]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Server(ServerOptions),
    Client(ClientArgs),
}

#[derive(Debug, Args, Clone)]
struct ServerOptions {
    #[arg(long)]
    ifname: String,

    #[arg(long, default_value_t = DEFAULT_PBPROBE_PORT)]
    port: u16,

    #[arg(long, value_enum, default_value_t = SendBackend::default())]
    send_backend: SendBackend,
}

#[derive(Debug, Clone)]
struct ClientOptions {
    ifname: String,
    peer: Ipv6Addr,
    config: PbProbeConfig,
}

#[derive(Debug, Args, Clone)]
struct ClientArgs {
    #[arg(long)]
    ifname: String,

    #[arg(long, value_parser = parse_link_local_addr_arg)]
    peer: Ipv6Addr,

    #[arg(long, default_value_t = DEFAULT_PBPROBE_PORT)]
    port: u16,

    #[arg(long = "samples", default_value_t = DEFAULT_SAMPLE_COUNT)]
    sample_count: u32,

    #[arg(long, default_value_t = DEFAULT_UTILIZATION)]
    utilization: f64,

    #[arg(long = "dispersion-threshold-ms", default_value_t = DEFAULT_DISPERSION_THRESHOLD_MS)]
    dispersion_threshold_ms: u64,

    #[arg(long = "initial-bulk-len", default_value_t = 1)]
    initial_bulk_len: u32,

    #[arg(long = "max-bulk-len", default_value_t = DEFAULT_MAX_BULK_LEN)]
    max_bulk_len: u32,

    #[arg(long = "ip-packet-bytes", default_value_t = usize::from(crate::config::PHYSICAL_LINK_MTU))]
    ip_packet_bytes: usize,

    #[arg(long = "start-timeout-ms", default_value_t = DEFAULT_START_TIMEOUT_MS)]
    start_timeout_ms: u64,

    #[arg(long = "rts-timeout-ms", default_value_t = DEFAULT_RTS_TIMEOUT_MS)]
    rts_timeout_ms: u64,

    #[arg(long = "control-retries", default_value_t = DEFAULT_CONTROL_RETRIES)]
    control_retries: u32,
}

impl ClientArgs {
    fn into_client_options(self) -> ClientOptions {
        ClientOptions {
            ifname: self.ifname,
            peer: self.peer,
            config: PbProbeConfig {
                port: self.port,
                sample_count: self.sample_count,
                utilization: self.utilization,
                dispersion_threshold: Duration::from_millis(self.dispersion_threshold_ms),
                initial_bulk_len: self.initial_bulk_len,
                max_bulk_len: self.max_bulk_len,
                ip_packet_bytes: self.ip_packet_bytes,
                start_timeout: Duration::from_millis(self.start_timeout_ms),
                rts_timeout: Duration::from_millis(self.rts_timeout_ms),
                control_retries: self.control_retries,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveSession {
    bulk_len: u32,
    sample_count: u32,
    ip_packet_bytes: u32,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum SendBackend {
    #[default]
    Auto,
    Batched,
    Connected,
}

fn run_server(options: ServerOptions) -> Result<()> {
    let (socket, ifindex) =
        open_link_local_udp(&options.ifname, options.port, Some(SERVER_RECV_TIMEOUT))
            .wrap_err_with(|| format!("opening PBProbe server on {}", options.ifname))?;
    let mut burst_sender = BurstSender::new(&options.ifname, &socket, options.send_backend)
        .wrap_err("initializing PBProbe UDP sender")?;
    let local_addr = socket
        .local_addr()
        .wrap_err("reading PBProbe server local address")?;

    println!(
        "PBProbe server on {} ifindex={} local={} port={}",
        options.ifname, ifindex, local_addr, options.port
    );

    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];
    let mut active = HashMap::<u64, ActiveSession>::new();

    loop {
        let (packet_len, from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) || err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving PBProbe packet"),
        };
        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok(header) = decode_header(packet) else {
            continue;
        };

        match header.kind {
            PacketKind::Start => {
                let session = ActiveSession {
                    bulk_len: header.bulk_len,
                    sample_count: header.sample_count,
                    ip_packet_bytes: header.ip_packet_bytes,
                };
                active.insert(header.run_id, session);
                send_header(
                    &socket,
                    from,
                    Header {
                        kind: PacketKind::StartAck,
                        ..header
                    },
                )?;
            }
            PacketKind::Rts => {
                let Some(session) = active.get(&header.run_id).copied() else {
                    continue;
                };
                if session.bulk_len != header.bulk_len
                    || session.sample_count != header.sample_count
                    || session.ip_packet_bytes != header.ip_packet_bytes
                {
                    continue;
                }
                send_bulk(&socket, &mut burst_sender, from, header, session)?;
            }
            PacketKind::End => {
                active.remove(&header.run_id);
            }
            PacketKind::StartAck
            | PacketKind::Bulk
            | PacketKind::Result
            | PacketKind::ErrorMessage => {}
        }
    }
}

fn run_client(options: ClientOptions) -> Result<()> {
    if options.config.sample_count == 0 {
        return Err(eyre!("sample count must be non-zero"));
    }
    if options.config.initial_bulk_len == 0 {
        return Err(eyre!("initial bulk length must be non-zero"));
    }
    if options.config.udp_payload_bytes() < HEADER_LEN {
        return Err(eyre!(
            "IP packet size {} leaves too little UDP payload for {HEADER_LEN}-byte PBProbe header",
            options.config.ip_packet_bytes
        ));
    }

    let (socket, ifindex) =
        open_link_local_udp(&options.ifname, 0, Some(options.config.start_timeout))
            .wrap_err_with(|| format!("opening PBProbe client on {}", options.ifname))?;
    let peer = scoped_peer_addr(options.peer, options.config.port, ifindex);
    let local_addr = socket
        .local_addr()
        .wrap_err("reading PBProbe client local address")?;

    println!(
        "PBProbe client on {} ifindex={} local={} peer={} port={}",
        options.ifname, ifindex, local_addr, options.peer, options.config.port
    );

    let estimate = run_probe(&socket, peer, &options.config)?;
    print_estimate(&estimate);
    Ok(())
}

fn run_probe(socket: &UdpSocket, peer: SocketAddr, config: &PbProbeConfig) -> Result<Estimate> {
    let mut bulk_len = config.initial_bulk_len;

    loop {
        let run_id = make_run_id();
        println!(
            "PBProbe round: run_id={run_id} bulk_len={bulk_len} samples={} utilization={:.4}",
            config.sample_count, config.utilization
        );
        start_session(socket, peer, run_id, bulk_len, config)?;
        let outcome = collect_estimate(socket, peer, run_id, bulk_len, config)?;
        send_end(socket, peer, run_id, bulk_len, config)?;

        match outcome {
            EstimateOutcome::Complete(estimate) => return Ok(estimate),
            EstimateOutcome::IncreaseBulk {
                previous_bulk_len,
                next_bulk_len,
                observed_dispersion,
            } => {
                println!(
                    "PBProbe increasing bulk length: {previous_bulk_len} -> {next_bulk_len}; observed dispersion {} below threshold {}",
                    format_duration(observed_dispersion),
                    format_duration(config.dispersion_threshold)
                );
                bulk_len = next_bulk_len;
            }
        }
    }
}

fn collect_estimate(
    socket: &UdpSocket,
    peer: SocketAddr,
    run_id: u64,
    bulk_len: u32,
    config: &PbProbeConfig,
) -> Result<EstimateOutcome> {
    let max_attempts = config
        .sample_count
        .saturating_mul(MIN_ATTEMPT_MULTIPLIER)
        .max(config.sample_count);
    let mut samples = Vec::with_capacity(usize::try_from(config.sample_count).unwrap_or(0));
    let mut attempts = 0_u32;
    let mut lost_samples = 0_u32;
    let mut min_dispersion: Option<Duration> = None;

    while u32::try_from(samples.len()).unwrap_or(u32::MAX) < config.sample_count {
        if attempts >= max_attempts {
            return Err(eyre!(
                "only collected {} accepted samples after {attempts} attempts",
                samples.len()
            ));
        }

        let sample_id = attempts;
        attempts = attempts.saturating_add(1);
        let started = Instant::now();
        send_rts(socket, peer, run_id, sample_id, bulk_len, config)?;

        let Some(sample) =
            receive_bulk_sample(socket, run_id, sample_id, bulk_len, started, config)?
        else {
            lost_samples = lost_samples.saturating_add(1);
            continue;
        };

        if sample.dispersion < config.dispersion_threshold
            && let Some(next) = next_bulk_len(bulk_len, config.max_bulk_len)
        {
            return Ok(EstimateOutcome::IncreaseBulk {
                previous_bulk_len: bulk_len,
                next_bulk_len: next,
                observed_dispersion: sample.dispersion,
            });
        }

        min_dispersion =
            Some(min_dispersion.map_or(sample.dispersion, |old| old.min(sample.dispersion)));
        samples.push(sample);

        if let Some(interval) =
            min_dispersion.and_then(|dispersion| pacing_interval(dispersion, config.utilization))
        {
            let elapsed = started.elapsed();
            if interval > elapsed {
                thread::sleep(interval - elapsed);
            }
        }
    }

    let selected = select_capacity_sample(&samples, config.ip_packet_bytes)
        .ok_or_else(|| eyre!("PBProbe collected no usable samples"))?;
    let sample_count = u32::try_from(samples.len()).unwrap_or(u32::MAX);
    let server_issue_samples = u32::try_from(
        samples
            .iter()
            .filter(|sample| sample.server_issue_duration.is_some())
            .count(),
    )
    .unwrap_or(u32::MAX);
    let min_server_issue_duration = samples
        .iter()
        .filter_map(|sample| sample.server_issue_duration)
        .min();
    Ok(EstimateOutcome::Complete(Estimate {
        bulk_len,
        sample_count,
        attempts,
        lost_samples,
        ip_packet_bytes: config.ip_packet_bytes,
        selected,
        min_dispersion: min_dispersion.unwrap_or(selected.sample.dispersion),
        server_issue_samples,
        min_server_issue_duration,
    }))
}

fn receive_bulk_sample(
    socket: &UdpSocket,
    run_id: u64,
    sample_id: u32,
    bulk_len: u32,
    started: Instant,
    config: &PbProbeConfig,
) -> Result<Option<AcceptedSample>> {
    let deadline = started + config.rts_timeout;
    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];
    let mut tracker = BulkSampleTracker::new(sample_id, bulk_len, started);

    loop {
        if !set_timeout_until(socket, deadline)? {
            return Ok(tracker.finish());
        }

        let (packet_len, _from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) => {
                return Ok(tracker.finish());
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving PBProbe bulk packet"),
        };
        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok((header, aux)) = decode_header_with_aux(packet) else {
            continue;
        };
        if header.kind != PacketKind::Bulk
            || header.run_id != run_id
            || header.sample_id != sample_id
            || header.bulk_len != bulk_len
            || header.seq > bulk_len
        {
            continue;
        }

        tracker.record(header.seq, Instant::now(), aux);
        if tracker.is_complete() {
            return Ok(tracker.finish());
        }
    }
}

#[derive(Debug)]
struct BulkSampleTracker {
    sample_id: u32,
    bulk_len: u32,
    started: Instant,
    next_seq: u32,
    first_rx: Option<Instant>,
    last_rx: Option<Instant>,
    server_issue_duration: Option<Duration>,
    valid: bool,
}

impl BulkSampleTracker {
    fn new(sample_id: u32, bulk_len: u32, started: Instant) -> Self {
        Self {
            sample_id,
            bulk_len,
            started,
            next_seq: 0,
            first_rx: None,
            last_rx: None,
            server_issue_duration: None,
            valid: true,
        }
    }

    fn record(&mut self, seq: u32, received: Instant, server_issue_nanos: u32) {
        if !self.valid || seq < self.next_seq {
            return;
        }
        if seq != self.next_seq {
            self.valid = false;
            return;
        }

        if seq == 0 {
            self.first_rx = Some(received);
        }
        if seq == self.bulk_len {
            self.last_rx = Some(received);
            if server_issue_nanos != 0 {
                self.server_issue_duration =
                    Some(Duration::from_nanos(u64::from(server_issue_nanos)));
            }
        }
        self.next_seq = self.next_seq.saturating_add(1);
    }

    fn is_complete(&self) -> bool {
        self.valid && self.next_seq == self.bulk_len.saturating_add(1)
    }

    fn finish(&self) -> Option<AcceptedSample> {
        if !self.is_complete() {
            return None;
        }
        let first = self.first_rx?;
        let last = self.last_rx?;
        if last < first {
            return None;
        }

        Some(AcceptedSample {
            sample_id: self.sample_id,
            bulk_len: self.bulk_len,
            delay_first: first.saturating_duration_since(self.started),
            delay_last: last.saturating_duration_since(self.started),
            dispersion: last.saturating_duration_since(first),
            server_issue_duration: self.server_issue_duration,
        })
    }
}

fn start_session(
    socket: &UdpSocket,
    peer: SocketAddr,
    run_id: u64,
    bulk_len: u32,
    config: &PbProbeConfig,
) -> Result<()> {
    let header = session_header(PacketKind::Start, run_id, bulk_len, config)?;
    let mut out = [0_u8; HEADER_LEN];
    encode_header(&mut out, header)?;

    for _attempt in 0..config.control_retries {
        send_datagram(socket, &out, peer)?;
        let deadline = Instant::now() + config.start_timeout;
        if wait_for_header(socket, deadline, |candidate| {
            candidate.kind == PacketKind::StartAck
                && candidate.run_id == run_id
                && candidate.bulk_len == bulk_len
        })?
        .is_some()
        {
            return Ok(());
        }
    }

    Err(eyre!("PBProbe peer did not acknowledge START"))
}

fn send_rts(
    socket: &UdpSocket,
    peer: SocketAddr,
    run_id: u64,
    sample_id: u32,
    bulk_len: u32,
    config: &PbProbeConfig,
) -> Result<()> {
    let mut header = session_header(PacketKind::Rts, run_id, bulk_len, config)?;
    header.sample_id = sample_id;
    send_header(socket, peer, header)
}

fn send_end(
    socket: &UdpSocket,
    peer: SocketAddr,
    run_id: u64,
    bulk_len: u32,
    config: &PbProbeConfig,
) -> Result<()> {
    send_header(
        socket,
        peer,
        session_header(PacketKind::End, run_id, bulk_len, config)?,
    )
}

fn send_bulk(
    socket: &UdpSocket,
    burst_sender: &mut BurstSender,
    peer: SocketAddr,
    request: Header,
    session: ActiveSession,
) -> Result<()> {
    let payload_bytes = udp_payload_bytes_from_ip(usize::try_from(session.ip_packet_bytes)?);
    if payload_bytes < HEADER_LEN {
        return Err(eyre!(
            "PBProbe packet size {} leaves too little UDP payload for {HEADER_LEN}-byte header",
            session.ip_packet_bytes
        ));
    }

    let packet_count = usize::try_from(session.bulk_len.saturating_add(1))
        .wrap_err("PBProbe bulk packet count exceeds usize")?;
    let burst_bytes = payload_bytes
        .checked_mul(packet_count)
        .ok_or_else(|| eyre!("PBProbe bulk buffer size overflow"))?;
    let mut out = vec![0_u8; burst_bytes];
    for seq in 0..=session.bulk_len {
        let seq_index = usize::try_from(seq).wrap_err("PBProbe sequence exceeds usize")?;
        let start = seq_index
            .checked_mul(payload_bytes)
            .ok_or_else(|| eyre!("PBProbe packet offset overflow"))?;
        let end = start
            .checked_add(payload_bytes)
            .ok_or_else(|| eyre!("PBProbe packet end overflow"))?;
        let Some(packet) = out.get_mut(start..end) else {
            return Err(eyre!("PBProbe packet slice out of bounds"));
        };
        encode_header_with_aux(
            packet,
            Header {
                kind: PacketKind::Bulk,
                run_id: request.run_id,
                sample_id: request.sample_id,
                seq,
                bulk_len: session.bulk_len,
                sample_count: session.sample_count,
                ip_packet_bytes: session.ip_packet_bytes,
            },
            0,
        )?;
    }

    let burst_started = Instant::now();
    burst_sender.send_bulk(socket, peer, &mut out, payload_bytes, |packet| {
        let clamped_issue_nanos =
            duration_nanos(burst_started.elapsed()).clamp(1, u64::from(u32::MAX));
        let server_issue_nanos = u32::try_from(clamped_issue_nanos).unwrap_or(u32::MAX);
        encode_header_with_aux(
            packet,
            Header {
                kind: PacketKind::Bulk,
                run_id: request.run_id,
                sample_id: request.sample_id,
                seq: session.bulk_len,
                bulk_len: session.bulk_len,
                sample_count: session.sample_count,
                ip_packet_bytes: session.ip_packet_bytes,
            },
            server_issue_nanos,
        )
    })?;
    let server_issue_duration = burst_started.elapsed();
    println!(
        "PBProbe server bulk: sample={} bulk_len={} issue={} issue_rate={}",
        request.sample_id,
        session.bulk_len,
        format_duration(server_issue_duration),
        format_optional_mbps(issue_rate_mbps(
            session.bulk_len.saturating_add(1),
            usize::try_from(session.ip_packet_bytes)?,
            server_issue_duration,
        ))
    );
    Ok(())
}

struct BurstSender {
    inner: BurstSenderInner,
}

enum BurstSenderInner {
    Batched {
        state: UdpSocketState,
    },
    Connected {
        socket: UdpSocket,
        connected_peer: Option<SocketAddr>,
    },
}

impl BurstSender {
    fn new(ifname: &str, socket: &UdpSocket, backend: SendBackend) -> io::Result<Self> {
        match backend {
            SendBackend::Connected => Self::new_connected(ifname),
            SendBackend::Batched => Self::new_batched(socket),
            SendBackend::Auto => {
                if cfg!(target_vendor = "apple") || cfg!(target_os = "macos") {
                    Self::new_connected(ifname)
                } else {
                    Self::new_batched(socket)
                }
            }
        }
    }

    fn new_batched(socket: &UdpSocket) -> io::Result<Self> {
        let state = UdpSocketState::new(UdpSockRef::from(socket))?;
        socket.set_nonblocking(false)?;
        Ok(Self {
            inner: BurstSenderInner::Batched { state },
        })
    }

    fn new_connected(ifname: &str) -> io::Result<Self> {
        let (socket, _ifindex) = open_link_local_udp(ifname, 0, None)?;
        Ok(Self {
            inner: BurstSenderInner::Connected {
                socket,
                connected_peer: None,
            },
        })
    }

    fn send_bulk<F>(
        &mut self,
        socket: &UdpSocket,
        peer: SocketAddr,
        buf: &mut [u8],
        datagram_bytes: usize,
        before_final_packet: F,
    ) -> Result<()>
    where
        F: FnMut(&mut [u8]) -> std::result::Result<usize, ProtocolError>,
    {
        if datagram_bytes == 0 {
            return Err(eyre!("PBProbe datagram size must be non-zero"));
        }

        match &mut self.inner {
            BurstSenderInner::Batched { state } => send_bulk_batched(
                state,
                socket,
                peer,
                buf,
                datagram_bytes,
                before_final_packet,
            ),
            BurstSenderInner::Connected {
                socket,
                connected_peer,
            } => send_bulk_connected(
                socket,
                connected_peer,
                peer,
                buf,
                datagram_bytes,
                before_final_packet,
            ),
        }
    }
}

fn send_bulk_batched<F>(
    state: &UdpSocketState,
    socket: &UdpSocket,
    peer: SocketAddr,
    buf: &mut [u8],
    datagram_bytes: usize,
    mut before_final_packet: F,
) -> Result<()>
where
    F: FnMut(&mut [u8]) -> std::result::Result<usize, ProtocolError>,
{
    let chunk_bytes = datagram_bytes
        .checked_mul(BATCH_SIZE)
        .ok_or_else(|| eyre!("PBProbe bulk chunk size overflow"))?;
    let total_bytes = buf.len();
    let mut sent_bytes = 0_usize;
    for chunk in buf.chunks_mut(chunk_bytes) {
        let next_sent_bytes = sent_bytes
            .checked_add(chunk.len())
            .ok_or_else(|| eyre!("PBProbe sent byte count overflow"))?;
        if next_sent_bytes == total_bytes {
            encode_final_packet(chunk, datagram_bytes, &mut before_final_packet)?;
        }
        let transmit = Transmit {
            destination: peer,
            ecn: None,
            contents: chunk,
            segment_size: Some(datagram_bytes),
            src_ip: None,
        };
        state
            .try_send(UdpSockRef::from(socket), &transmit)
            .wrap_err_with(|| format!("sending PBProbe bulk to {peer}"))?;
        sent_bytes = next_sent_bytes;
    }

    Ok(())
}

fn send_bulk_connected<F>(
    socket: &UdpSocket,
    connected_peer: &mut Option<SocketAddr>,
    peer: SocketAddr,
    buf: &mut [u8],
    datagram_bytes: usize,
    mut before_final_packet: F,
) -> Result<()>
where
    F: FnMut(&mut [u8]) -> std::result::Result<usize, ProtocolError>,
{
    if *connected_peer != Some(peer) {
        socket
            .connect(peer)
            .wrap_err_with(|| format!("connecting PBProbe data socket to {peer}"))?;
        *connected_peer = Some(peer);
    }

    let total_bytes = buf.len();
    let mut sent_bytes = 0_usize;
    for packet in buf.chunks_mut(datagram_bytes) {
        let next_sent_bytes = sent_bytes
            .checked_add(packet.len())
            .ok_or_else(|| eyre!("PBProbe sent byte count overflow"))?;
        if next_sent_bytes == total_bytes {
            before_final_packet(packet)?;
        }
        let sent = socket
            .send(packet)
            .wrap_err_with(|| format!("sending PBProbe connected bulk to {peer}"))?;
        if sent != packet.len() {
            return Err(eyre!(
                "short UDP send to {peer}: sent {sent} of {} bytes",
                packet.len()
            ));
        }
        sent_bytes = next_sent_bytes;
    }

    Ok(())
}

fn encode_final_packet<F>(
    chunk: &mut [u8],
    datagram_bytes: usize,
    before_final_packet: &mut F,
) -> Result<()>
where
    F: FnMut(&mut [u8]) -> std::result::Result<usize, ProtocolError>,
{
    let final_packet_start = chunk
        .len()
        .checked_sub(datagram_bytes)
        .ok_or_else(|| eyre!("PBProbe final packet chunk underflow"))?;
    let Some(final_packet) = chunk.get_mut(final_packet_start..) else {
        return Err(eyre!("PBProbe final packet slice out of bounds"));
    };
    before_final_packet(final_packet)?;
    Ok(())
}

fn session_header(
    kind: PacketKind,
    run_id: u64,
    bulk_len: u32,
    config: &PbProbeConfig,
) -> Result<Header> {
    Ok(Header {
        kind,
        run_id,
        sample_id: 0,
        seq: 0,
        bulk_len,
        sample_count: config.sample_count,
        ip_packet_bytes: u32::try_from(config.ip_packet_bytes)
            .wrap_err("PBProbe IP packet size exceeds u32")?,
    })
}

fn send_header(socket: &UdpSocket, peer: SocketAddr, header: Header) -> Result<()> {
    let mut out = [0_u8; HEADER_LEN];
    encode_header(&mut out, header)?;
    send_datagram(socket, &out, peer)
}

fn wait_for_header<F>(
    socket: &UdpSocket,
    deadline: Instant,
    mut matches: F,
) -> Result<Option<Header>>
where
    F: FnMut(Header) -> bool,
{
    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];

    loop {
        if !set_timeout_until(socket, deadline)? {
            return Ok(None);
        }

        let (packet_len, _from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) => return Ok(None),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving PBProbe control packet"),
        };
        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok(header) = decode_header(packet) else {
            continue;
        };
        if matches(header) {
            return Ok(Some(header));
        }
    }
}

fn send_datagram(socket: &UdpSocket, buf: &[u8], target: SocketAddr) -> Result<()> {
    let sent = socket
        .send_to(buf, target)
        .wrap_err_with(|| format!("sending PBProbe packet to {target}"))?;
    if sent != buf.len() {
        return Err(eyre!(
            "short UDP send to {target}: sent {sent} of {} bytes",
            buf.len()
        ));
    }
    Ok(())
}

fn set_timeout_until(socket: &UdpSocket, deadline: Instant) -> Result<bool> {
    let now = Instant::now();
    if now >= deadline {
        return Ok(false);
    }
    socket
        .set_read_timeout(Some(deadline.saturating_duration_since(now)))
        .wrap_err("setting PBProbe socket read timeout")?;
    Ok(true)
}

fn udp_payload_bytes_from_ip(ip_packet_bytes: usize) -> usize {
    let overhead = usize::from(OUTER_IPV6_HEADER_BYTES + OUTER_UDP_HEADER_BYTES);
    ip_packet_bytes.saturating_sub(overhead)
}

fn print_estimate(estimate: &Estimate) {
    let server_issue = estimate.selected.sample.server_issue_duration;
    let server_issue_rate = server_issue.and_then(|duration| {
        issue_rate_mbps(
            estimate.bulk_len.saturating_add(1),
            estimate.ip_packet_bytes,
            duration,
        )
    });
    let min_server_issue_rate = estimate.min_server_issue_duration.and_then(|duration| {
        issue_rate_mbps(
            estimate.bulk_len.saturating_add(1),
            estimate.ip_packet_bytes,
            duration,
        )
    });
    println!(
        "PBProbe estimate: {:.1} Mbps bulk_len={} samples={} attempts={} lost_samples={} selected_sample={} delay_sum={} dispersion={} min_dispersion={} selected_server_issue={} selected_server_issue_rate={} server_issue_samples={} min_server_issue={} min_server_issue_rate={}",
        estimate.selected.capacity_mbps,
        estimate.bulk_len,
        estimate.sample_count,
        estimate.attempts,
        estimate.lost_samples,
        estimate.selected.sample.sample_id,
        format_duration(estimate.selected.sample.delay_sum()),
        format_duration(estimate.selected.sample.dispersion),
        format_duration(estimate.min_dispersion),
        format_optional_duration(server_issue),
        format_optional_mbps(server_issue_rate),
        estimate.server_issue_samples,
        format_optional_duration(estimate.min_server_issue_duration),
        format_optional_mbps(min_server_issue_rate)
    );
}

fn issue_rate_mbps(packet_count: u32, ip_packet_bytes: usize, duration: Duration) -> Option<f64> {
    let nanos = duration.as_nanos();
    if packet_count == 0 || ip_packet_bytes == 0 || nanos == 0 {
        return None;
    }

    let bits = f64::from(packet_count) * (ip_packet_bytes as f64) * 8.0;
    Some(bits * 1_000.0 / (nanos as f64))
}

fn parse_link_local_addr_arg(raw: &str) -> std::result::Result<Ipv6Addr, String> {
    parse_link_local_addr(raw).map_err(|err| err.to_string())
}

fn is_timeout(err: &io::Error) -> bool {
    matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

fn make_run_id() -> u64 {
    rand::random()
}

fn format_duration(duration: Duration) -> String {
    if duration < Duration::from_millis(1) {
        return format!("{:.3} us", duration.as_secs_f64() * 1_000_000.0);
    }
    format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
}

fn format_optional_duration(duration: Option<Duration>) -> String {
    duration
        .map(format_duration)
        .unwrap_or_else(|| "n/a".to_owned())
}

fn format_optional_mbps(mbps: Option<f64>) -> String {
    mbps.map(|value| format!("{value:.1} Mbps"))
        .unwrap_or_else(|| "n/a".to_owned())
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;
    use std::time::{Duration, Instant};

    use clap::Parser;

    use super::{BulkSampleTracker, Cli, Command};

    #[test]
    fn parses_server_options() {
        let cli = Cli::try_parse_from([
            "pbprobe_link",
            "server",
            "--ifname",
            "en3",
            "--port",
            "42001",
        ])
        .expect("server options should parse");

        match cli.command {
            Command::Server(options) => {
                assert_eq!(options.ifname, "en3");
                assert_eq!(options.port, 42_001);
            }
            Command::Client(_) => panic!("expected server command"),
        }
    }

    #[test]
    fn parses_client_options() {
        let cli = Cli::try_parse_from([
            "pbprobe_link",
            "client",
            "--ifname",
            "en2",
            "--peer",
            "fe80::1%en2",
            "--samples",
            "50",
            "--initial-bulk-len",
            "100",
        ])
        .expect("client options should parse");

        match cli.command {
            Command::Client(args) => {
                let options = args.into_client_options();
                assert_eq!(options.ifname, "en2");
                assert_eq!(
                    options.peer,
                    "fe80::1".parse::<Ipv6Addr>().expect("valid IPv6")
                );
                assert_eq!(options.config.sample_count, 50);
                assert_eq!(options.config.initial_bulk_len, 100);
            }
            Command::Server(_) => panic!("expected client command"),
        }
    }

    #[test]
    fn bulk_tracker_accepts_complete_ordered_bulk() {
        let started = Instant::now();
        let mut tracker = BulkSampleTracker::new(7, 2, started);

        tracker.record(0, started + Duration::from_millis(1), 0);
        tracker.record(1, started + Duration::from_millis(2), 0);
        tracker.record(2, started + Duration::from_millis(3), 1000);

        let sample = tracker.finish().expect("complete bulk should be accepted");
        assert_eq!(sample.sample_id, 7);
        assert_eq!(sample.bulk_len, 2);
        assert_eq!(sample.delay_first, Duration::from_millis(1));
        assert_eq!(sample.delay_last, Duration::from_millis(3));
        assert_eq!(sample.dispersion, Duration::from_millis(2));
        assert_eq!(sample.server_issue_duration, Some(Duration::from_micros(1)));
    }

    #[test]
    fn bulk_tracker_rejects_incomplete_bulk() {
        let started = Instant::now();
        let mut tracker = BulkSampleTracker::new(7, 2, started);

        tracker.record(0, started + Duration::from_millis(1), 0);
        tracker.record(2, started + Duration::from_millis(3), 1000);

        assert!(tracker.finish().is_none());
    }

    #[test]
    fn bulk_tracker_rejects_out_of_order_bulk() {
        let started = Instant::now();
        let mut tracker = BulkSampleTracker::new(7, 2, started);

        tracker.record(1, started + Duration::from_millis(2), 0);
        tracker.record(0, started + Duration::from_millis(1), 0);
        tracker.record(2, started + Duration::from_millis(3), 1000);

        assert!(tracker.finish().is_none());
    }
}
