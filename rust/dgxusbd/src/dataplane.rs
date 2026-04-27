use std::io::ErrorKind;
use std::os::fd::AsRawFd as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, Context as _, OptionExt as _};
use crossbeam_channel::{Receiver, Sender, bounded};
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use nusb::transfer::{Bulk, In, Out, TransferError};

use crate::bridge::{BridgeCounters, BridgeOptions, BridgeReport};
use crate::ncm::{
    DEFAULT_NTB_MAX_SIZE, ETHERNET_HEADER_LEN, NcmError, NtbBuildConfig, NtbParseConfig,
};
use crate::tap::create_tap;
use crate::usb::{NcmPair, NcmSetupReport, open_ncm_pair};

const TOKEN_TAP: Token = Token(0);
const TAP_POLL_INTERVAL: Duration = Duration::from_millis(250);
const SUPERVISOR_POLL_INTERVAL: Duration = Duration::from_millis(20);

struct BridgeDataplane {
    tap_name: String,
    pair: NcmPair,
    setup_report: NcmSetupReport,
    shared: Arc<SharedDataplaneState>,
    errors: Receiver<String>,
    tap_to_usb: Option<JoinHandle<()>>,
    usb_to_tap: Option<JoinHandle<()>>,
}

struct SharedDataplaneState {
    stop: AtomicBool,
    remaining_events: Option<AtomicU64>,
    counters: AtomicBridgeCounters,
}

#[derive(Default)]
struct AtomicBridgeCounters {
    tap_frames_rx: AtomicU64,
    tap_frames_dropped: AtomicU64,
    usb_ntbs_tx: AtomicU64,
    usb_ntbs_rx: AtomicU64,
    usb_timeouts: AtomicU64,
    usb_frames_rx: AtomicU64,
    tap_frames_tx: AtomicU64,
    tap_write_dropped: AtomicU64,
    malformed_ntbs: AtomicU64,
}

struct TapToUsbWorker {
    tap: tun_rs::SyncDevice,
    ep_out: nusb::Endpoint<Bulk, Out>,
    build_config: NtbBuildConfig,
    write_timeout: Duration,
    tap_buffer: Vec<u8>,
    batch_frames: Vec<Vec<u8>>,
    batch_limit: usize,
    shared: Arc<SharedDataplaneState>,
    sequence: u16,
}

struct UsbToTapWorker {
    tap: tun_rs::SyncDevice,
    ep_in: nusb::Endpoint<Bulk, In>,
    parse_config: NtbParseConfig,
    read_size: usize,
    read_timeout: Duration,
    read_queue_depth: usize,
    drain_budget_ntbs: usize,
    shared: Arc<SharedDataplaneState>,
}

pub fn run_bridge_dataplane(options: BridgeOptions) -> eyre::Result<BridgeReport> {
    BridgeDataplane::start(options)?.wait()
}

impl BridgeDataplane {
    fn start(options: BridgeOptions) -> eyre::Result<Self> {
        let open = open_ncm_pair(options.open)?;
        let bulk_in = open
            .pair
            .bulk_in
            .ok_or_eyre("selected pair has no bulk IN endpoint")?;
        let bulk_out = open
            .pair
            .bulk_out
            .ok_or_eyre("selected pair has no bulk OUT endpoint")?;
        let ep_in = open
            .data_interface
            .endpoint::<Bulk, In>(bulk_in)
            .wrap_err_with(|| format!("failed to open bulk IN endpoint {bulk_in:#04x}"))?;
        let ep_out = open
            .data_interface
            .endpoint::<Bulk, Out>(bulk_out)
            .wrap_err_with(|| format!("failed to open bulk OUT endpoint {bulk_out:#04x}"))?;

        let mut tap_options = options.tap.clone();
        tap_options.nonblocking = true;
        let tap = create_tap(&tap_options)?;
        let tap_tx = tap
            .device
            .try_clone()
            .wrap_err("failed to clone TAP handle for USB-to-TAP worker")?;
        tap_tx
            .set_nonblocking(true)
            .wrap_err("failed to put cloned TAP handle in nonblocking mode")?;

        let parse_config = open
            .setup_report
            .ntb_parameters
            .map_or_else(NtbParseConfig::default, NtbParseConfig::from);
        let build_config = open
            .setup_report
            .ntb_parameters
            .map_or_else(NtbBuildConfig::default, NtbBuildConfig::from);

        let read_size = transfer_size(parse_config.max_size, ep_in.max_packet_size());
        let batch_limit = batch_limit(
            options.tap_budget_frames.max(1),
            usize::from(tap.mtu) + ETHERNET_HEADER_LEN,
            build_config,
        );
        let shared = Arc::new(SharedDataplaneState::new(options.max_events));
        let (errors_send, errors) = bounded(2);

        let tap_worker = TapToUsbWorker {
            tap: tap.device,
            ep_out,
            build_config,
            write_timeout: options.usb_write_timeout,
            tap_buffer: vec![0; usize::from(tap.mtu) + ETHERNET_HEADER_LEN + 64],
            batch_frames: Vec::with_capacity(batch_limit),
            batch_limit,
            shared: Arc::clone(&shared),
            sequence: 0,
        };
        let usb_worker = UsbToTapWorker {
            tap: tap_tx,
            ep_in,
            parse_config,
            read_size,
            read_timeout: options.usb_read_timeout,
            read_queue_depth: options.usb_read_queue_depth.max(1),
            drain_budget_ntbs: options.usb_budget_ntbs.max(1),
            shared: Arc::clone(&shared),
        };

        let tap_to_usb = spawn_worker("dgxusbd-tap-to-usb", errors_send.clone(), move || {
            tap_worker.run()
        })?;
        let usb_to_tap = spawn_worker("dgxusbd-usb-to-tap", errors_send, move || usb_worker.run())?;

        let dataplane = Self {
            tap_name: tap.name,
            pair: open.pair,
            setup_report: open.setup_report,
            shared,
            errors,
            tap_to_usb: Some(tap_to_usb),
            usb_to_tap: Some(usb_to_tap),
        };
        dataplane.wait_until_done(options.duration)?;
        Ok(dataplane)
    }

    fn wait(mut self) -> eyre::Result<BridgeReport> {
        self.stop_and_join()?;
        Ok(BridgeReport {
            tap_name: self.tap_name.clone(),
            pair: self.pair.clone(),
            setup_report: self.setup_report.clone(),
            counters: self.shared.counters.snapshot(),
        })
    }

    fn wait_until_done(&self, duration: Option<Duration>) -> eyre::Result<()> {
        let deadline = duration.map(|duration| Instant::now() + duration);
        loop {
            if let Ok(message) = self.errors.try_recv() {
                self.shared.stop();
                return Err(eyre::eyre!(message));
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                self.shared.stop();
                return Ok(());
            }
            if self.shared.event_budget_exhausted() {
                self.shared.stop();
                return Ok(());
            }
            thread::sleep(SUPERVISOR_POLL_INTERVAL);
        }
    }

    fn stop_and_join(&mut self) -> eyre::Result<()> {
        self.shared.stop();
        let tap_result = join_worker(self.tap_to_usb.take(), "TAP-to-USB worker");
        let usb_result = join_worker(self.usb_to_tap.take(), "USB-to-TAP worker");
        tap_result.and(usb_result)
    }
}

impl Drop for BridgeDataplane {
    fn drop(&mut self) {
        let _: eyre::Result<()> = self.stop_and_join();
    }
}

impl SharedDataplaneState {
    fn new(max_events: Option<u64>) -> Self {
        Self {
            stop: AtomicBool::new(false),
            remaining_events: max_events.map(AtomicU64::new),
            counters: AtomicBridgeCounters::default(),
        }
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Acquire) || self.event_budget_exhausted()
    }

    fn event_budget_exhausted(&self) -> bool {
        self.remaining_events
            .as_ref()
            .is_some_and(|remaining| remaining.load(Ordering::Acquire) == 0)
    }

    fn record_total_event(&self) -> bool {
        if self.stop.load(Ordering::Acquire) {
            return false;
        }
        let Some(remaining) = self.remaining_events.as_ref() else {
            return true;
        };
        let mut current = remaining.load(Ordering::Acquire);
        loop {
            if current == 0 {
                self.stop();
                return false;
            }
            match remaining.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
    }

    fn inc_tap_frames_rx(&self) -> bool {
        if self.record_total_event() {
            self.counters.tap_frames_rx.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn inc_usb_ntbs_tx(&self) -> bool {
        if self.record_total_event() {
            self.counters.usb_ntbs_tx.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn inc_usb_ntbs_rx(&self) -> bool {
        if self.record_total_event() {
            self.counters.usb_ntbs_rx.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn inc_malformed_ntbs(&self) -> bool {
        if self.record_total_event() {
            self.counters.malformed_ntbs.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

impl AtomicBridgeCounters {
    fn snapshot(&self) -> BridgeCounters {
        BridgeCounters {
            tap_frames_rx: self.tap_frames_rx.load(Ordering::Relaxed),
            tap_frames_dropped: self.tap_frames_dropped.load(Ordering::Relaxed),
            usb_ntbs_tx: self.usb_ntbs_tx.load(Ordering::Relaxed),
            usb_ntbs_rx: self.usb_ntbs_rx.load(Ordering::Relaxed),
            usb_timeouts: self.usb_timeouts.load(Ordering::Relaxed),
            usb_frames_rx: self.usb_frames_rx.load(Ordering::Relaxed),
            tap_frames_tx: self.tap_frames_tx.load(Ordering::Relaxed),
            tap_write_dropped: self.tap_write_dropped.load(Ordering::Relaxed),
            malformed_ntbs: self.malformed_ntbs.load(Ordering::Relaxed),
        }
    }
}

impl TapToUsbWorker {
    fn run(mut self) -> eyre::Result<()> {
        let mut poll = Poll::new().wrap_err("creating TAP poller")?;
        let mut events = Events::with_capacity(8);
        let tap_raw_fd = self.tap.as_raw_fd();
        let mut tap_source = SourceFd(&tap_raw_fd);
        poll.registry()
            .register(&mut tap_source, TOKEN_TAP, Interest::READABLE)
            .wrap_err("registering TAP fd with poller")?;

        while !self.shared.should_stop() {
            poll.poll(&mut events, Some(TAP_POLL_INTERVAL))
                .wrap_err("polling TAP fd")?;
            for event in &events {
                if event.token() == TOKEN_TAP {
                    self.drain_tap_ready()?;
                }
            }
        }

        Ok(())
    }

    fn drain_tap_ready(&mut self) -> eyre::Result<()> {
        self.batch_frames.clear();
        while self.batch_frames.len() < self.batch_limit && !self.shared.should_stop() {
            match self.tap.recv(&mut self.tap_buffer) {
                Ok(length) => {
                    if !self.shared.inc_tap_frames_rx() {
                        return Ok(());
                    }
                    if length < ETHERNET_HEADER_LEN {
                        self.shared
                            .counters
                            .tap_frames_dropped
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let frame = self
                        .tap_buffer
                        .get(..length)
                        .ok_or_eyre("TAP read length exceeded TAP buffer")?;
                    self.batch_frames.push(frame.to_vec());
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err).wrap_err("failed to read TAP frame"),
            }
        }

        if self.batch_frames.is_empty() {
            return Ok(());
        }
        let frames = std::mem::take(&mut self.batch_frames);
        self.send_batch(&frames)?;
        self.batch_frames = frames;
        Ok(())
    }

    fn send_batch(&mut self, frames: &[Vec<u8>]) -> eyre::Result<()> {
        if frames.is_empty() || self.shared.should_stop() {
            return Ok(());
        }
        let refs: Vec<_> = frames.iter().map(Vec::as_slice).collect();
        match crate::ncm::build_ntb16(self.sequence, &refs, self.build_config) {
            Ok(ntb) => self.send_ntb(ntb),
            Err(NcmError::BuiltNtbTooLarge { .. }) if frames.len() > 1 => {
                let mid = frames.len() / 2;
                let (left, right) = frames.split_at(mid);
                self.send_batch(left)?;
                self.send_batch(right)
            }
            Err(err) => Err(err).wrap_err("failed to build NTB16 from TAP frame batch"),
        }
    }

    fn send_ntb(&mut self, ntb: Vec<u8>) -> eyre::Result<()> {
        if !self.shared.inc_usb_ntbs_tx() {
            return Ok(());
        }
        self.ep_out
            .transfer_blocking(ntb.into(), self.write_timeout)
            .status
            .map_err(usb_transfer_error)
            .wrap_err("failed to write NTB16 to USB OUT endpoint")?;
        self.sequence = self.sequence.wrapping_add(1);
        Ok(())
    }
}

impl UsbToTapWorker {
    fn run(mut self) -> eyre::Result<()> {
        self.fill_read_queue();
        while !self.shared.should_stop() {
            for _ in 0..self.drain_budget_ntbs {
                if self.shared.should_stop() {
                    break;
                }
                let Some(completion) = self.ep_in.wait_next_complete(self.read_timeout) else {
                    self.shared
                        .counters
                        .usb_timeouts
                        .fetch_add(1, Ordering::Relaxed);
                    break;
                };
                let buffer = completion.buffer;
                match completion.status {
                    Ok(()) => {
                        if !self.shared.inc_usb_ntbs_rx() {
                            return Ok(());
                        }
                        let ntb = buffer
                            .get(..completion.actual_len)
                            .ok_or_eyre("USB completion length exceeded transfer buffer")?;
                        self.handle_ntb(ntb)?;
                    }
                    Err(TransferError::Cancelled) if self.shared.should_stop() => return Ok(()),
                    Err(TransferError::Cancelled) => {
                        self.shared
                            .counters
                            .usb_timeouts
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        return Err(usb_transfer_error(err))
                            .wrap_err("failed to read USB IN endpoint");
                    }
                }
                if !self.shared.should_stop() {
                    self.ep_in.submit(buffer);
                }
            }
            self.fill_read_queue();
        }
        self.ep_in.cancel_all();
        Ok(())
    }

    fn fill_read_queue(&mut self) {
        while self.ep_in.pending() < self.read_queue_depth && !self.shared.should_stop() {
            let buffer = self.ep_in.allocate(self.read_size);
            self.ep_in.submit(buffer);
        }
    }

    fn handle_ntb(&self, ntb: &[u8]) -> eyre::Result<()> {
        match crate::ncm::parse_ntb16(ntb, self.parse_config) {
            Ok(parsed) => {
                for frame in parsed.frames {
                    match self.tap.send(frame) {
                        Ok(_) => {
                            self.shared
                                .counters
                                .usb_frames_rx
                                .fetch_add(1, Ordering::Relaxed);
                            self.shared
                                .counters
                                .tap_frames_tx
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(err) if err.kind() == ErrorKind::WouldBlock => {
                            self.shared
                                .counters
                                .tap_write_dropped
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Err(err) => return Err(err).wrap_err("failed to write frame to TAP"),
                    }
                }
            }
            Err(err) => {
                if self.shared.inc_malformed_ntbs() {
                    tracing::warn!(%err, "dropping malformed NTB16");
                }
            }
        }
        Ok(())
    }
}

fn spawn_worker(
    name: &'static str,
    errors: Sender<String>,
    work: impl FnOnce() -> eyre::Result<()> + Send + 'static,
) -> eyre::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            if let Err(err) = work() {
                let _: Result<(), _> = errors.try_send(format!("{name} failed: {err:?}"));
            }
        })
        .wrap_err_with(|| format!("spawning {name}"))
}

fn join_worker(handle: Option<JoinHandle<()>>, label: &str) -> eyre::Result<()> {
    if let Some(handle) = handle {
        handle.join().map_err(|_| eyre::eyre!("{label} panicked"))?;
    }
    Ok(())
}

fn transfer_size(configured_size: usize, max_packet_size: usize) -> usize {
    let capped = configured_size.clamp(max_packet_size.max(1), usize::from(u16::MAX));
    let packet = max_packet_size.max(1);
    let rounded = capped / packet * packet;
    rounded.max(packet).min(DEFAULT_NTB_MAX_SIZE.max(packet))
}

fn batch_limit(configured_limit: usize, frame_capacity: usize, config: NtbBuildConfig) -> usize {
    let frame_budget = frame_capacity
        .saturating_add(config.datagram_alignment.max(1))
        .saturating_add(8)
        .max(1);
    let size_budget = config
        .max_size
        .saturating_sub(64)
        .checked_div(frame_budget)
        .unwrap_or(1)
        .max(1);
    configured_limit
        .max(1)
        .min(config.max_datagrams.max(1))
        .min(size_budget)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_budget_is_exact_across_total_event_counters() {
        let shared = SharedDataplaneState::new(Some(3));

        assert!(shared.inc_tap_frames_rx());
        assert!(shared.inc_usb_ntbs_tx());
        assert!(shared.inc_usb_ntbs_rx());
        assert!(!shared.inc_malformed_ntbs());

        let counters = shared.counters.snapshot();
        assert_eq!(counters.total_events(), 3);
        assert_eq!(counters.malformed_ntbs, 0);
        assert!(shared.event_budget_exhausted());
    }
}
