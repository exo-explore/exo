use std::collections::{HashMap, HashSet};
use std::env;
use std::io::{self, ErrorKind};
use std::net::{Ipv6Addr, SocketAddr, UdpSocket};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{Result, WrapErr, eyre};

use crate::config::{OUTER_IPV6_HEADER_BYTES, OUTER_UDP_HEADER_BYTES};

use super::estimator::{
    AcceptedSample, Estimate, EstimateOutcome, PbProbeConfig, next_bulk_len, pacing_interval,
    select_capacity_sample,
};
use super::protocol::{
    HEADER_LEN, Header, PacketKind, decode_header, decode_header_with_aux, duration_nanos,
    encode_header, encode_header_with_aux,
};
use crate::profiling::socket::{open_link_local_udp, parse_link_local_addr, scoped_peer_addr};

const MAX_UDP_PACKET_BYTES: usize = 65_535;
const SERVER_RECV_TIMEOUT: Duration = Duration::from_secs(1);
const MIN_ATTEMPT_MULTIPLIER: u32 = 3;

pub fn run_from_env() -> Result<()> {
    run(env::args())
}

pub fn run<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<String>>();
    if wants_help(&args) {
        print_usage();
        return Ok(());
    }

    let command = args
        .get(1)
        .map(String::as_str)
        .ok_or_else(|| eyre!("missing command; expected `server` or `client`"))?;
    let rest = args.get(2..).unwrap_or_default();

    match command {
        "server" => run_server(parse_server_options(rest)?),
        "client" => run_client(parse_client_options(rest)?),
        other => Err(eyre!(
            "unknown command {other:?}; expected `server` or `client`"
        )),
    }
}

#[derive(Debug, Clone)]
struct ServerOptions {
    ifname: String,
    port: u16,
}

#[derive(Debug, Clone)]
struct ClientOptions {
    ifname: String,
    peer: Ipv6Addr,
    config: PbProbeConfig,
}

#[derive(Debug, Clone, Copy)]
struct ActiveSession {
    bulk_len: u32,
    sample_count: u32,
    ip_packet_bytes: u32,
}

fn run_server(options: ServerOptions) -> Result<()> {
    let (socket, ifindex) =
        open_link_local_udp(&options.ifname, options.port, Some(SERVER_RECV_TIMEOUT))
            .wrap_err_with(|| format!("opening PBProbe server on {}", options.ifname))?;
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
                send_bulk(&socket, from, header, session)?;
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
    let mut round = 0_u64;

    loop {
        let run_id = make_run_id(round);
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
                round = round.saturating_add(1);
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
    let mut seen = HashSet::with_capacity(usize::try_from(bulk_len.saturating_add(1)).unwrap_or(0));
    let mut first_rx = None;
    let mut last_rx = None;
    let mut server_issue_duration = None;

    loop {
        if !set_timeout_until(socket, deadline)? {
            return Ok(finish_sample(
                sample_id,
                bulk_len,
                started,
                first_rx,
                last_rx,
                server_issue_duration,
            ));
        }

        let (packet_len, _from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) => {
                return Ok(finish_sample(
                    sample_id,
                    bulk_len,
                    started,
                    first_rx,
                    last_rx,
                    server_issue_duration,
                ));
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

        let now = Instant::now();
        seen.insert(header.seq);
        if header.seq == 0 {
            first_rx = Some(now);
        }
        if header.seq == bulk_len {
            last_rx = Some(now);
            if aux != 0 {
                server_issue_duration = Some(Duration::from_nanos(u64::from(aux)));
            }
        }

        if seen.len() == usize::try_from(bulk_len.saturating_add(1)).unwrap_or(usize::MAX) {
            return Ok(finish_sample(
                sample_id,
                bulk_len,
                started,
                first_rx,
                last_rx,
                server_issue_duration,
            ));
        }
    }
}

fn finish_sample(
    sample_id: u32,
    bulk_len: u32,
    started: Instant,
    first_rx: Option<Instant>,
    last_rx: Option<Instant>,
    server_issue_duration: Option<Duration>,
) -> Option<AcceptedSample> {
    let first = first_rx?;
    let last = last_rx?;
    if last < first {
        return None;
    }

    Some(AcceptedSample {
        sample_id,
        bulk_len,
        delay_first: first.saturating_duration_since(started),
        delay_last: last.saturating_duration_since(started),
        dispersion: last.saturating_duration_since(first),
        server_issue_duration,
    })
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

    let mut out = vec![0_u8; payload_bytes];
    let burst_started = Instant::now();
    for seq in 0..=session.bulk_len {
        let server_issue_nanos = if seq == session.bulk_len {
            duration_nanos(burst_started.elapsed()).min(u64::from(u32::MAX)) as u32
        } else {
            0
        };
        encode_header_with_aux(
            &mut out,
            Header {
                kind: PacketKind::Bulk,
                run_id: request.run_id,
                sample_id: request.sample_id,
                seq,
                bulk_len: session.bulk_len,
                sample_count: session.sample_count,
                ip_packet_bytes: session.ip_packet_bytes,
            },
            server_issue_nanos,
        )?;
        send_datagram(socket, &out, peer)?;
    }
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

fn parse_server_options(args: &[String]) -> Result<ServerOptions> {
    let mut flags = parse_flags(args)?;
    let ifname = take_required(&mut flags, "ifname")?;
    let port = take_u16(&mut flags, "port")?.unwrap_or(super::estimator::DEFAULT_PBPROBE_PORT);
    reject_unknown_flags(flags)?;
    Ok(ServerOptions { ifname, port })
}

fn parse_client_options(args: &[String]) -> Result<ClientOptions> {
    let mut flags = parse_flags(args)?;
    let ifname = take_required(&mut flags, "ifname")?;
    let peer_raw = take_required(&mut flags, "peer")?;
    let peer = parse_link_local_addr(&peer_raw)
        .wrap_err_with(|| format!("parsing peer IPv6 address {peer_raw:?}"))?;

    let mut config = PbProbeConfig::default();
    config.port = take_u16(&mut flags, "port")?.unwrap_or(config.port);
    config.sample_count = take_u32(&mut flags, "samples")?.unwrap_or(config.sample_count);
    config.utilization = take_f64(&mut flags, "utilization")?.unwrap_or(config.utilization);
    config.dispersion_threshold = take_duration_ms(&mut flags, "dispersion-threshold-ms")?
        .unwrap_or(config.dispersion_threshold);
    config.initial_bulk_len =
        take_u32(&mut flags, "initial-bulk-len")?.unwrap_or(config.initial_bulk_len);
    config.max_bulk_len = take_u32(&mut flags, "max-bulk-len")?.unwrap_or(config.max_bulk_len);
    config.ip_packet_bytes =
        take_usize(&mut flags, "ip-packet-bytes")?.unwrap_or(config.ip_packet_bytes);
    config.start_timeout =
        take_duration_ms(&mut flags, "start-timeout-ms")?.unwrap_or(config.start_timeout);
    config.rts_timeout =
        take_duration_ms(&mut flags, "rts-timeout-ms")?.unwrap_or(config.rts_timeout);
    config.control_retries =
        take_u32(&mut flags, "control-retries")?.unwrap_or(config.control_retries);

    reject_unknown_flags(flags)?;
    Ok(ClientOptions {
        ifname,
        peer,
        config,
    })
}

fn parse_flags(args: &[String]) -> Result<HashMap<String, String>> {
    let mut flags = HashMap::new();
    let mut index = 0;

    while let Some(raw) = args.get(index) {
        let Some(without_prefix) = raw.strip_prefix("--") else {
            return Err(eyre!("expected --flag value, got {raw:?}"));
        };

        if let Some((key, value)) = without_prefix.split_once('=') {
            flags.insert(key.to_owned(), value.to_owned());
            index += 1;
            continue;
        }

        let Some(value) = args.get(index + 1) else {
            return Err(eyre!("missing value for --{without_prefix}"));
        };
        if value.starts_with("--") {
            return Err(eyre!("missing value for --{without_prefix}"));
        }
        flags.insert(without_prefix.to_owned(), value.to_owned());
        index += 2;
    }

    Ok(flags)
}

fn take_required(flags: &mut HashMap<String, String>, name: &str) -> Result<String> {
    flags
        .remove(name)
        .ok_or_else(|| eyre!("missing required --{name}"))
}

fn take_u16(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<u16>> {
    take_parse(flags, name)
}

fn take_u32(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<u32>> {
    take_parse(flags, name)
}

fn take_usize(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<usize>> {
    take_parse(flags, name)
}

fn take_f64(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<f64>> {
    take_parse(flags, name)
}

fn take_duration_ms(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<Duration>> {
    let Some(raw) = flags.remove(name) else {
        return Ok(None);
    };
    let millis = raw
        .parse::<u64>()
        .wrap_err_with(|| format!("parsing --{name} value {raw:?} as milliseconds"))?;
    Ok(Some(Duration::from_millis(millis)))
}

fn take_parse<T>(flags: &mut HashMap<String, String>, name: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let Some(raw) = flags.remove(name) else {
        return Ok(None);
    };
    raw.parse::<T>()
        .map(Some)
        .wrap_err_with(|| format!("parsing --{name} value {raw:?}"))
}

fn reject_unknown_flags(flags: HashMap<String, String>) -> Result<()> {
    if flags.is_empty() {
        return Ok(());
    }

    let mut names = flags.keys().map(String::as_str).collect::<Vec<&str>>();
    names.sort_unstable();
    Err(eyre!("unknown option(s): --{}", names.join(", --")))
}

fn wants_help(args: &[String]) -> bool {
    args.len() <= 1
        || args
            .iter()
            .any(|arg| arg.as_str() == "--help" || arg.as_str() == "-h")
}

fn is_timeout(err: &io::Error) -> bool {
    matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

fn make_run_id(round: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, duration_nanos);
    now ^ (u64::from(std::process::id()) << 32) ^ round
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

fn print_usage() {
    println!(
        "\
Usage:
  cargo run -p babblerd --example pbprobe_link -- server --ifname IFACE [--port PORT]
  cargo run -p babblerd --example pbprobe_link -- client --ifname IFACE --peer FE80::ADDR [options]

Client options:
  --port PORT                       server UDP port, default 41902
  --samples N                       fixed accepted sample count, default 200
  --utilization FLOAT               PBProbe utilization cap U, default 0.01
  --dispersion-threshold-ms MS      D_thresh for bulk growth, default 1
  --initial-bulk-len K              starting k, default 1
  --max-bulk-len K                  maximum k, default 10000
  --ip-packet-bytes N               modeled IPv6 packet size, default 1500
  --start-timeout-ms MS             START/ACK timeout, default 750
  --rts-timeout-ms MS               per-bulk timeout, default 750
  --control-retries N               START retries, default 5
"
    );
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;

    use super::{parse_client_options, parse_server_options};

    #[test]
    fn parses_server_options() {
        let args = vec![
            "--ifname".to_owned(),
            "en3".to_owned(),
            "--port".to_owned(),
            "42001".to_owned(),
        ];

        let options = parse_server_options(&args).expect("server options should parse");
        assert_eq!(options.ifname, "en3");
        assert_eq!(options.port, 42_001);
    }

    #[test]
    fn parses_client_options() {
        let args = vec![
            "--ifname".to_owned(),
            "en2".to_owned(),
            "--peer".to_owned(),
            "fe80::1%en2".to_owned(),
            "--samples".to_owned(),
            "50".to_owned(),
            "--initial-bulk-len".to_owned(),
            "100".to_owned(),
        ];

        let options = parse_client_options(&args).expect("client options should parse");
        assert_eq!(options.ifname, "en2");
        assert_eq!(
            options.peer,
            "fe80::1".parse::<Ipv6Addr>().expect("valid IPv6")
        );
        assert_eq!(options.config.sample_count, 50);
        assert_eq!(options.config.initial_bulk_len, 100);
    }
}
