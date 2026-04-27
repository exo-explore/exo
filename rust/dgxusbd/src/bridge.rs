use std::fmt::Write as _;
use std::time::Duration;

use color_eyre::eyre;

use crate::tap::TapOptions;
use crate::usb::{NcmPair, NcmSetupReport, OpenPairOptions, render_setup_report};

pub const DEFAULT_BRIDGE_USB_TIMEOUT: Duration = Duration::from_millis(100);
pub const DEFAULT_BRIDGE_USB_READ_QUEUE_DEPTH: usize = 8;

#[derive(Clone, Debug)]
pub struct BridgeOptions {
    pub open: OpenPairOptions,
    pub tap: TapOptions,
    pub duration: Option<Duration>,
    pub max_events: Option<u64>,
    pub usb_read_timeout: Duration,
    pub usb_write_timeout: Duration,
    pub tap_budget_frames: usize,
    pub usb_budget_ntbs: usize,
    pub usb_read_queue_depth: usize,
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
    pub tap_write_dropped: u64,
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
    crate::dataplane::run_bridge_dataplane(options)
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
        "counters: tap_rx={} tap_drop={} usb_tx_ntb={} usb_rx_ntb={} usb_timeout={} usb_rx_frames={} tap_tx={} tap_tx_drop={} malformed_ntb={}",
        report.counters.tap_frames_rx,
        report.counters.tap_frames_dropped,
        report.counters.usb_ntbs_tx,
        report.counters.usb_ntbs_rx,
        report.counters.usb_timeouts,
        report.counters.usb_frames_rx,
        report.counters.tap_frames_tx,
        report.counters.tap_write_dropped,
        report.counters.malformed_ntbs
    );
    out
}

impl BridgeCounters {
    #[must_use]
    pub fn total_events(&self) -> u64 {
        self.tap_frames_rx + self.usb_ntbs_rx + self.usb_ntbs_tx + self.malformed_ntbs
    }
}
