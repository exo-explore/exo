use std::fmt::Write as _;
use std::io::ErrorKind;
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, Context as _, OptionExt as _};
use nusb::transfer::{Bulk, In, Out, TransferError};

use crate::ncm::{DEFAULT_NTB_MAX_SIZE, ETHERNET_HEADER_LEN, NtbBuildConfig, NtbParseConfig};
use crate::tap::{TapOptions, create_tap};
use crate::usb::{NcmPair, NcmSetupReport, OpenPairOptions, open_ncm_pair, render_setup_report};

pub const DEFAULT_BRIDGE_USB_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Clone, Debug)]
pub struct BridgeOptions {
    pub open: OpenPairOptions,
    pub tap: TapOptions,
    pub duration: Option<Duration>,
    pub max_events: Option<u64>,
    pub usb_timeout: Duration,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BridgeCounters {
    pub tap_frames_rx: u64,
    pub tap_frames_dropped: u64,
    pub usb_ntbs_tx: u64,
    pub usb_ntbs_rx: u64,
    pub usb_timeouts: u64,
    pub usb_frames_rx: u64,
    pub tap_frames_tx: u64,
    pub malformed_ntbs: u64,
}

#[derive(Clone, Debug)]
pub struct BridgeReport {
    pub tap_name: String,
    pub pair: NcmPair,
    pub setup_report: NcmSetupReport,
    pub counters: BridgeCounters,
}

pub fn run_bridge(options: BridgeOptions) -> eyre::Result<BridgeReport> {
    let open = open_ncm_pair(options.open)?;
    let bulk_in = open
        .pair
        .bulk_in
        .ok_or_eyre("selected pair has no bulk IN endpoint")?;
    let bulk_out = open
        .pair
        .bulk_out
        .ok_or_eyre("selected pair has no bulk OUT endpoint")?;
    let mut ep_in = open
        .data_interface
        .endpoint::<Bulk, In>(bulk_in)
        .wrap_err_with(|| format!("failed to open bulk IN endpoint {bulk_in:#04x}"))?;
    let mut ep_out = open
        .data_interface
        .endpoint::<Bulk, Out>(bulk_out)
        .wrap_err_with(|| format!("failed to open bulk OUT endpoint {bulk_out:#04x}"))?;

    let mut tap_options = options.tap.clone();
    tap_options.nonblocking = true;
    let tap = create_tap(&tap_options)?;

    let parse_config = open
        .setup_report
        .ntb_parameters
        .map_or_else(NtbParseConfig::default, NtbParseConfig::from);
    let build_config = open
        .setup_report
        .ntb_parameters
        .map_or_else(NtbBuildConfig::default, NtbBuildConfig::from);
    let read_size = transfer_size(parse_config.max_size, ep_in.max_packet_size());

    let deadline = options.duration.map(|duration| Instant::now() + duration);
    let mut counters = BridgeCounters::default();
    let mut tap_buffer = vec![0; usize::from(tap.mtu) + ETHERNET_HEADER_LEN + 64];
    let mut sequence = 0_u16;

    loop {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }
        if options
            .max_events
            .is_some_and(|max_events| total_events(&counters) >= max_events)
        {
            break;
        }

        drain_tap_to_usb(
            &tap.device,
            &mut tap_buffer,
            &mut ep_out,
            build_config,
            options.usb_timeout,
            &mut sequence,
            &mut counters,
        )?;
        poll_usb_to_tap(
            &tap.device,
            &mut ep_in,
            parse_config,
            read_size,
            options.usb_timeout,
            &mut counters,
        )?;
    }

    Ok(BridgeReport {
        tap_name: tap.name,
        pair: open.pair,
        setup_report: open.setup_report,
        counters,
    })
}

#[must_use]
pub fn render_bridge_report(report: &BridgeReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "bridge: tap={}", report.tap_name);
    let _ = writeln!(
        out,
        "pair: control if{} -> data if{} in={:#04x} out={:#04x}",
        report.pair.control_interface,
        report.pair.data_interface,
        report.pair.bulk_in.unwrap_or_default(),
        report.pair.bulk_out.unwrap_or_default()
    );
    render_setup_report(&mut out, &report.setup_report);
    let _ = writeln!(
        out,
        "counters: tap_rx={} tap_drop={} usb_tx_ntb={} usb_rx_ntb={} usb_timeout={} usb_rx_frames={} tap_tx={} malformed_ntb={}",
        report.counters.tap_frames_rx,
        report.counters.tap_frames_dropped,
        report.counters.usb_ntbs_tx,
        report.counters.usb_ntbs_rx,
        report.counters.usb_timeouts,
        report.counters.usb_frames_rx,
        report.counters.tap_frames_tx,
        report.counters.malformed_ntbs
    );
    out
}

fn drain_tap_to_usb(
    tap: &tun_rs::SyncDevice,
    tap_buffer: &mut [u8],
    ep_out: &mut nusb::Endpoint<Bulk, Out>,
    build_config: NtbBuildConfig,
    usb_timeout: Duration,
    sequence: &mut u16,
    counters: &mut BridgeCounters,
) -> eyre::Result<()> {
    loop {
        match tap.recv(tap_buffer) {
            Ok(length) => {
                counters.tap_frames_rx += 1;
                if length < ETHERNET_HEADER_LEN {
                    counters.tap_frames_dropped += 1;
                    continue;
                }
                let frame = &tap_buffer[..length];
                let ntb = crate::ncm::build_ntb16(*sequence, &[frame], build_config)
                    .wrap_err("failed to build NTB16 from TAP frame")?;
                *sequence = sequence.wrapping_add(1);
                ep_out
                    .transfer_blocking(ntb.into(), usb_timeout)
                    .status
                    .map_err(usb_transfer_error)
                    .wrap_err("failed to write NTB16 to USB OUT endpoint")?;
                counters.usb_ntbs_tx += 1;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => return Ok(()),
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err).wrap_err("failed to read TAP frame"),
        }
    }
}

fn poll_usb_to_tap(
    tap: &tun_rs::SyncDevice,
    ep_in: &mut nusb::Endpoint<Bulk, In>,
    parse_config: NtbParseConfig,
    read_size: usize,
    usb_timeout: Duration,
    counters: &mut BridgeCounters,
) -> eyre::Result<()> {
    let completion = ep_in.transfer_blocking(nusb::transfer::Buffer::new(read_size), usb_timeout);
    match completion.status {
        Ok(()) => {}
        Err(TransferError::Cancelled) => {
            counters.usb_timeouts += 1;
            return Ok(());
        }
        Err(err) => return Err(usb_transfer_error(err)).wrap_err("failed to read USB IN endpoint"),
    }

    counters.usb_ntbs_rx += 1;
    let ntb = &completion.buffer[..completion.actual_len];
    match crate::ncm::parse_ntb16(ntb, parse_config) {
        Ok(parsed) => {
            for frame in parsed.frames {
                tap.send(frame).wrap_err("failed to write frame to TAP")?;
                counters.usb_frames_rx += 1;
                counters.tap_frames_tx += 1;
            }
        }
        Err(err) => {
            counters.malformed_ntbs += 1;
            tracing::warn!(%err, "dropping malformed NTB16");
        }
    }

    Ok(())
}

fn transfer_size(configured_size: usize, max_packet_size: usize) -> usize {
    let capped = configured_size.clamp(max_packet_size.max(1), u16::MAX as usize);
    let packet = max_packet_size.max(1);
    let rounded = capped / packet * packet;
    rounded.max(packet).min(DEFAULT_NTB_MAX_SIZE.max(packet))
}

fn total_events(counters: &BridgeCounters) -> u64 {
    counters.tap_frames_rx + counters.usb_ntbs_rx + counters.usb_ntbs_tx + counters.malformed_ntbs
}

fn usb_transfer_error(err: TransferError) -> eyre::Report {
    eyre::eyre!("{err}")
}

impl From<crate::usb::NtbParametersReport> for NtbParseConfig {
    fn from(parameters: crate::usb::NtbParametersReport) -> Self {
        parameters.rx_parse_config()
    }
}

impl From<crate::usb::NtbParametersReport> for NtbBuildConfig {
    fn from(parameters: crate::usb::NtbParametersReport) -> Self {
        parameters.tx_build_config()
    }
}
