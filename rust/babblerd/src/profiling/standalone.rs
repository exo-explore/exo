use std::collections::{HashMap, HashSet};
use std::env;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, UdpSocket};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{Result, WrapErr, eyre};

use super::estimator::{CapacitySample, duration_nanos_u64, latency_stats};
use super::protocol::{
    HEADER_LEN, Header, PacketKind, SUMMARY_PACKET_LEN, SummaryBody, decode_header,
    decode_summary_body, encode_header, encode_summary,
};
use super::socket::{
    open_link_local_udp, parse_link_local_addr, scoped_peer_addr, with_default_scope,
};
use super::types::{DEFAULT_PROFILE_PORT, ProbeConfig};

const MAX_UDP_PACKET_BYTES: usize = 65_535;
const REFLECT_RECV_TIMEOUT: Duration = Duration::from_secs(1);
const STALE_TRAIN_AFTER: Duration = Duration::from_secs(60);
const SUMMARY_REQUEST_ATTEMPTS: u32 = 3;

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
        .ok_or_else(|| eyre!("missing command; expected `probe` or `reflect`"))?;
    let rest = args.get(2..).unwrap_or_default();

    match command {
        "probe" => run_probe(parse_probe_options(rest)?),
        "reflect" => run_reflect(parse_reflect_options(rest)?),
        other => Err(eyre!(
            "unknown command {other:?}; expected `probe` or `reflect`"
        )),
    }
}

#[derive(Debug, Clone)]
struct ProbeOptions {
    ifname: String,
    peer: std::net::Ipv6Addr,
    config: ProbeConfig,
}

#[derive(Debug, Clone)]
struct ReflectOptions {
    ifname: String,
    port: u16,
}

#[derive(Debug)]
struct TrainAccumulator {
    received_packets: u32,
    received_bytes: u64,
    first_rx: Option<Instant>,
    last_rx: Option<Instant>,
    last_update: Instant,
    seen: HashSet<u32>,
}

impl TrainAccumulator {
    fn new(expected_packets: u32, now: Instant) -> Self {
        Self {
            received_packets: 0,
            received_bytes: 0,
            first_rx: None,
            last_rx: None,
            last_update: now,
            seen: HashSet::with_capacity(usize::try_from(expected_packets).unwrap_or(0)),
        }
    }

    fn record(&mut self, seq: u32, packet_len: usize, now: Instant) {
        self.last_update = now;
        if !self.seen.insert(seq) {
            return;
        }

        self.received_packets = self.received_packets.saturating_add(1);
        self.received_bytes = self
            .received_bytes
            .saturating_add(u64::try_from(packet_len).unwrap_or(u64::MAX));
        if self.first_rx.is_none() {
            self.first_rx = Some(now);
        }
        self.last_rx = Some(now);
    }

    fn summary(&self) -> SummaryBody {
        let span_nanos = match (self.first_rx, self.last_rx) {
            (Some(first), Some(last)) => duration_nanos_u64(last.saturating_duration_since(first)),
            _ => 0,
        };

        SummaryBody {
            received_packets: self.received_packets,
            received_bytes: self.received_bytes,
            span_nanos,
        }
    }
}

fn run_reflect(options: ReflectOptions) -> Result<()> {
    let (socket, ifindex) =
        open_link_local_udp(&options.ifname, options.port, Some(REFLECT_RECV_TIMEOUT))
            .wrap_err_with(|| format!("opening profiling reflector on {}", options.ifname))?;
    let local_addr = socket
        .local_addr()
        .wrap_err("reading reflector local address")?;

    println!(
        "reflecting profiling probes on {} ifindex={} local={}",
        options.ifname, ifindex, local_addr
    );

    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];
    let mut trains = HashMap::<u64, TrainAccumulator>::new();
    let mut last_cleanup = Instant::now();

    loop {
        cleanup_stale_trains(&mut trains, &mut last_cleanup);

        let (packet_len, from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) || err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving profiling packet"),
        };

        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok(header) = decode_header(packet) else {
            continue;
        };

        match header.kind {
            PacketKind::EchoRequest => {
                send_echo_reply(&socket, ifindex, from, header).wrap_err("sending echo reply")?;
            }
            PacketKind::Train => {
                let now = Instant::now();
                trains
                    .entry(header.run_id)
                    .or_insert_with(|| TrainAccumulator::new(header.count, now))
                    .record(header.seq, packet_len, now);
            }
            PacketKind::SummaryRequest => {
                send_summary_reply(&socket, ifindex, from, header, &mut trains)
                    .wrap_err("sending train summary")?;
            }
            PacketKind::EchoReply | PacketKind::SummaryReply => {}
        }
    }
}

fn run_probe(options: ProbeOptions) -> Result<()> {
    if options.config.train_payload_bytes < HEADER_LEN {
        return Err(eyre!(
            "train payload must be at least {HEADER_LEN} bytes, got {}",
            options.config.train_payload_bytes
        ));
    }
    if options.config.train_packets == 0 {
        return Err(eyre!("train packet count must be non-zero"));
    }

    let (socket, ifindex) =
        open_link_local_udp(&options.ifname, 0, Some(options.config.echo_timeout))
            .wrap_err_with(|| format!("opening profiling probe socket on {}", options.ifname))?;
    let peer = scoped_peer_addr(options.peer, options.config.port, ifindex);
    let local_addr = socket
        .local_addr()
        .wrap_err("reading probe local address")?;
    let base_run_id = make_base_run_id();

    println!(
        "probing {} via {} ifindex={} local={} peer_port={}",
        options.peer, options.ifname, ifindex, local_addr, options.config.port
    );
    println!(
        "capacity probe: rounds={} train_packets={} payload_bytes={} interval_ms={}",
        options.config.capacity_rounds,
        options.config.train_packets,
        options.config.train_payload_bytes,
        options.config.train_interval.as_millis()
    );

    let latency_samples = run_echo_probes(&socket, peer, base_run_id, &options.config)
        .wrap_err("running latency probes")?;
    print_latency_summary(options.config.echo_count, &latency_samples);

    let capacity_samples = run_capacity_probes(&socket, peer, base_run_id, &options.config)
        .wrap_err("running capacity probes")?;
    print_capacity_summary(&capacity_samples);

    Ok(())
}

fn send_echo_reply(
    socket: &UdpSocket,
    ifindex: u32,
    from: SocketAddr,
    request: Header,
) -> Result<()> {
    let reply = Header {
        kind: PacketKind::EchoReply,
        run_id: request.run_id,
        seq: request.seq,
        count: request.count,
    };
    let mut out = [0_u8; HEADER_LEN];
    encode_header(&mut out, reply)?;
    send_datagram(socket, &out, with_default_scope(from, ifindex))
}

fn send_summary_reply(
    socket: &UdpSocket,
    ifindex: u32,
    from: SocketAddr,
    request: Header,
    trains: &mut HashMap<u64, TrainAccumulator>,
) -> Result<()> {
    let body = trains
        .get(&request.run_id)
        .map_or_else(empty_summary, TrainAccumulator::summary);
    let reply = Header {
        kind: PacketKind::SummaryReply,
        run_id: request.run_id,
        seq: 0,
        count: request.count,
    };
    let mut out = [0_u8; SUMMARY_PACKET_LEN];
    let len = encode_summary(&mut out, reply, body)?;
    send_datagram(
        socket,
        out.get(..len).unwrap_or(&out),
        with_default_scope(from, ifindex),
    )?;
    trains.remove(&request.run_id);
    Ok(())
}

fn run_echo_probes(
    socket: &UdpSocket,
    peer: SocketAddr,
    base_run_id: u64,
    config: &ProbeConfig,
) -> Result<Vec<Duration>> {
    let mut samples = Vec::new();
    let run_id = base_run_id ^ 0xe0c0_u64;

    for seq in 0..config.echo_count {
        let header = Header {
            kind: PacketKind::EchoRequest,
            run_id,
            seq,
            count: config.echo_count,
        };
        let mut out = [0_u8; HEADER_LEN];
        encode_header(&mut out, header)?;

        let start = Instant::now();
        send_datagram(socket, &out, peer)?;
        match receive_echo_reply(socket, run_id, seq, start, config.echo_timeout)? {
            Some(sample) => {
                println!("echo {:>3}: {}", seq + 1, format_duration(sample));
                samples.push(sample);
            }
            None => {
                println!("echo {:>3}: timeout", seq + 1);
            }
        }

        thread::sleep(config.echo_interval);
    }

    Ok(samples)
}

fn run_capacity_probes(
    socket: &UdpSocket,
    peer: SocketAddr,
    base_run_id: u64,
    config: &ProbeConfig,
) -> Result<Vec<CapacitySample>> {
    let mut samples = Vec::new();
    let mut train = vec![0_u8; config.train_payload_bytes];

    for round in 0..config.capacity_rounds {
        let run_id = base_run_id ^ (0xc0_ffee_u64.wrapping_add(u64::from(round)));
        let sender_start = Instant::now();

        for seq in 0..config.train_packets {
            let header = Header {
                kind: PacketKind::Train,
                run_id,
                seq,
                count: config.train_packets,
            };
            encode_header(&mut train, header)?;
            send_datagram(socket, &train, peer)?;
        }

        let sender_span = sender_start.elapsed();
        thread::sleep(config.train_settle);

        let summary = request_summary(socket, peer, run_id, config)?;
        match summary {
            Some(body) => {
                let sample = CapacitySample {
                    sent_packets: config.train_packets,
                    received_packets: body.received_packets,
                    received_bytes: body.received_bytes,
                    span: Duration::from_nanos(body.span_nanos),
                };
                print_capacity_round(round + 1, sample, sender_span);
                samples.push(sample);
            }
            None => {
                println!(
                    "capacity {:>3}: summary timeout after sender_burst={}",
                    round + 1,
                    format_duration(sender_span)
                );
            }
        }

        thread::sleep(config.train_interval);
    }

    Ok(samples)
}

fn request_summary(
    socket: &UdpSocket,
    peer: SocketAddr,
    run_id: u64,
    config: &ProbeConfig,
) -> Result<Option<SummaryBody>> {
    let request = Header {
        kind: PacketKind::SummaryRequest,
        run_id,
        seq: 0,
        count: config.train_packets,
    };
    let mut out = [0_u8; HEADER_LEN];
    encode_header(&mut out, request)?;

    let attempt_timeout = div_duration(config.echo_timeout, SUMMARY_REQUEST_ATTEMPTS);
    for _attempt in 0..SUMMARY_REQUEST_ATTEMPTS {
        send_datagram(socket, &out, peer)?;
        let deadline = Instant::now() + attempt_timeout;
        if let Some(summary) = receive_summary_reply(socket, run_id, deadline)? {
            return Ok(Some(summary));
        }
    }

    Ok(None)
}

fn receive_echo_reply(
    socket: &UdpSocket,
    run_id: u64,
    seq: u32,
    start: Instant,
    timeout: Duration,
) -> Result<Option<Duration>> {
    let deadline = start + timeout;
    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];

    loop {
        if !set_timeout_until(socket, deadline)? {
            return Ok(None);
        }

        let (packet_len, _from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) => return Ok(None),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving echo reply"),
        };
        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok(header) = decode_header(packet) else {
            continue;
        };
        if header.kind == PacketKind::EchoReply && header.run_id == run_id && header.seq == seq {
            return Ok(Some(start.elapsed()));
        }
    }
}

fn receive_summary_reply(
    socket: &UdpSocket,
    run_id: u64,
    deadline: Instant,
) -> Result<Option<SummaryBody>> {
    let mut buf = vec![0_u8; MAX_UDP_PACKET_BYTES];

    loop {
        if !set_timeout_until(socket, deadline)? {
            return Ok(None);
        }

        let (packet_len, _from) = match socket.recv_from(&mut buf) {
            Ok(received) => received,
            Err(err) if is_timeout(&err) => return Ok(None),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("receiving train summary"),
        };
        let Some(packet) = buf.get(..packet_len) else {
            continue;
        };
        let Ok(header) = decode_header(packet) else {
            continue;
        };
        if header.kind == PacketKind::SummaryReply && header.run_id == run_id {
            return Ok(Some(decode_summary_body(packet)?));
        }
    }
}

fn send_datagram(socket: &UdpSocket, buf: &[u8], target: SocketAddr) -> Result<()> {
    let sent = socket
        .send_to(buf, target)
        .wrap_err_with(|| format!("sending profiling packet to {target}"))?;
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
        .wrap_err("setting profiling socket read timeout")?;
    Ok(true)
}

fn cleanup_stale_trains(trains: &mut HashMap<u64, TrainAccumulator>, last_cleanup: &mut Instant) {
    if last_cleanup.elapsed() < Duration::from_secs(5) {
        return;
    }

    trains.retain(|_run_id, train| train.last_update.elapsed() < STALE_TRAIN_AFTER);
    *last_cleanup = Instant::now();
}

fn empty_summary() -> SummaryBody {
    SummaryBody {
        received_packets: 0,
        received_bytes: 0,
        span_nanos: 0,
    }
}

fn print_latency_summary(sent: u32, samples: &[Duration]) {
    match latency_stats(sent, samples) {
        Some(stats) => println!(
            "latency summary: sent={} received={} loss={:.1}% min={} avg={} max={}",
            stats.sent,
            stats.received,
            stats.loss_ratio * 100.0,
            format_duration(stats.min),
            format_duration(stats.avg),
            format_duration(stats.max)
        ),
        None => println!("latency summary: no replies"),
    }
}

fn print_capacity_round(round: u32, sample: CapacitySample, sender_span: Duration) {
    let mbps = sample
        .mbps()
        .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.1} Mbps"));
    println!(
        "capacity {:>3}: rx={}/{} loss={:.1}% span={} estimate={} sender_burst={}",
        round,
        sample.received_packets,
        sample.sent_packets,
        sample.loss_ratio() * 100.0,
        format_duration(sample.span),
        mbps,
        format_duration(sender_span)
    );
}

fn print_capacity_summary(samples: &[CapacitySample]) {
    let mut estimates = samples
        .iter()
        .filter_map(|sample| sample.mbps())
        .collect::<Vec<f64>>();
    if estimates.is_empty() {
        println!("capacity summary: no usable samples");
        return;
    }

    estimates.sort_by(f64::total_cmp);
    let median_index = estimates.len() / 2;
    let median = estimates.get(median_index).copied().unwrap_or(0.0);
    let best = estimates.last().copied().unwrap_or(median);
    let received = samples
        .iter()
        .map(|sample| sample.received_packets)
        .sum::<u32>();
    let sent = samples
        .iter()
        .map(|sample| sample.sent_packets)
        .sum::<u32>();
    let loss = if sent == 0 {
        0.0
    } else {
        f64::from(sent.saturating_sub(received)) / f64::from(sent)
    };

    println!(
        "capacity summary: samples={} median={median:.1} Mbps best={best:.1} Mbps aggregate_loss={:.1}%",
        estimates.len(),
        loss * 100.0
    );
}

fn parse_probe_options(args: &[String]) -> Result<ProbeOptions> {
    let mut flags = parse_flags(args)?;
    let ifname = take_required(&mut flags, "ifname")?;
    let peer_raw = take_required(&mut flags, "peer")?;
    let peer = parse_link_local_addr(&peer_raw)
        .wrap_err_with(|| format!("parsing peer IPv6 address {peer_raw:?}"))?;

    let mut config = ProbeConfig::default();
    config.port = take_u16(&mut flags, "port")?.unwrap_or(DEFAULT_PROFILE_PORT);
    config.echo_count = take_u32(&mut flags, "echo-count")?.unwrap_or(config.echo_count);
    config.echo_interval =
        take_duration_ms(&mut flags, "echo-interval-ms")?.unwrap_or(config.echo_interval);
    config.echo_timeout =
        take_duration_ms(&mut flags, "timeout-ms")?.unwrap_or(config.echo_timeout);
    config.capacity_rounds =
        take_u32(&mut flags, "capacity-rounds")?.unwrap_or(config.capacity_rounds);
    config.train_packets = take_u32(&mut flags, "train-packets")?.unwrap_or(config.train_packets);
    config.train_payload_bytes =
        take_usize(&mut flags, "payload-bytes")?.unwrap_or(config.train_payload_bytes);
    config.train_interval =
        take_duration_ms(&mut flags, "train-interval-ms")?.unwrap_or(config.train_interval);
    config.train_settle = take_duration_ms(&mut flags, "settle-ms")?.unwrap_or(config.train_settle);

    reject_unknown_flags(flags)?;

    Ok(ProbeOptions {
        ifname,
        peer,
        config,
    })
}

fn parse_reflect_options(args: &[String]) -> Result<ReflectOptions> {
    let mut flags = parse_flags(args)?;
    let ifname = take_required(&mut flags, "ifname")?;
    let port = take_u16(&mut flags, "port")?.unwrap_or(DEFAULT_PROFILE_PORT);
    reject_unknown_flags(flags)?;
    Ok(ReflectOptions { ifname, port })
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

fn div_duration(duration: Duration, divisor: u32) -> Duration {
    if divisor == 0 {
        return duration;
    }
    Duration::from_nanos(duration_nanos_u64(duration) / u64::from(divisor))
}

fn format_duration(duration: Duration) -> String {
    if duration < Duration::from_millis(1) {
        return format!("{:.3} us", duration.as_secs_f64() * 1_000_000.0);
    }
    format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
}

fn make_base_run_id() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, duration_nanos_u64);
    now ^ (u64::from(std::process::id()) << 32)
}

fn print_usage() {
    println!(
        "\
Usage:
  cargo run -p babblerd --example link_profile -- reflect --ifname IFACE [--port PORT]
  cargo run -p babblerd --example link_profile -- probe --ifname IFACE --peer FE80::ADDR [options]

Probe options:
  --port PORT                 reflector UDP port, default {DEFAULT_PROFILE_PORT}
  --echo-count N              latency probes, default 10
  --echo-interval-ms MS       delay between latency probes, default 250
  --timeout-ms MS             receive timeout, default 500
  --capacity-rounds N         packet-train rounds, default 5
  --train-packets N           MTU-sized packets per round, default 64
  --payload-bytes N           UDP payload bytes per train packet, default 1452
  --train-interval-ms MS      delay between train rounds, default 1000
  --settle-ms MS              delay before summary request, default 25
"
    );
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;

    use super::{parse_probe_options, parse_reflect_options};

    #[test]
    fn parses_probe_options() {
        let args = vec![
            "--ifname".to_owned(),
            "en3".to_owned(),
            "--peer".to_owned(),
            "fe80::1%en3".to_owned(),
            "--train-packets".to_owned(),
            "32".to_owned(),
        ];

        let options = parse_probe_options(&args).expect("probe options should parse");
        assert_eq!(options.ifname, "en3");
        assert_eq!(
            options.peer,
            "fe80::1".parse::<Ipv6Addr>().expect("valid IPv6")
        );
        assert_eq!(options.config.train_packets, 32);
    }

    #[test]
    fn parses_reflect_options() {
        let args = vec![
            "--ifname".to_owned(),
            "en2".to_owned(),
            "--port".to_owned(),
            "42000".to_owned(),
        ];

        let options = parse_reflect_options(&args).expect("reflect options should parse");
        assert_eq!(options.ifname, "en2");
        assert_eq!(options.port, 42_000);
    }
}
