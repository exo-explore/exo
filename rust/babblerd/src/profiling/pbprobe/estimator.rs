use std::time::Duration;

use crate::config::{OUTER_IPV6_HEADER_BYTES, OUTER_UDP_HEADER_BYTES, PHYSICAL_LINK_MTU};

pub const DEFAULT_PBPROBE_PORT: u16 = 41_902;
pub const DEFAULT_SAMPLE_COUNT: u32 = 200;
pub const DEFAULT_UTILIZATION: f64 = 0.01;
pub const DEFAULT_DISPERSION_THRESHOLD: Duration = Duration::from_millis(1);
pub const DEFAULT_MAX_BULK_LEN: u32 = 10_000;
pub const DEFAULT_RTS_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_START_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_CONTROL_RETRIES: u32 = 5;

#[derive(Debug, Clone)]
pub struct PbProbeConfig {
    pub port: u16,
    pub sample_count: u32,
    pub utilization: f64,
    pub dispersion_threshold: Duration,
    pub initial_bulk_len: u32,
    pub max_bulk_len: u32,
    pub ip_packet_bytes: usize,
    pub start_timeout: Duration,
    pub rts_timeout: Duration,
    pub control_retries: u32,
}

impl Default for PbProbeConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_PBPROBE_PORT,
            sample_count: DEFAULT_SAMPLE_COUNT,
            utilization: DEFAULT_UTILIZATION,
            dispersion_threshold: DEFAULT_DISPERSION_THRESHOLD,
            initial_bulk_len: 1,
            max_bulk_len: DEFAULT_MAX_BULK_LEN,
            ip_packet_bytes: usize::from(PHYSICAL_LINK_MTU),
            start_timeout: DEFAULT_START_TIMEOUT,
            rts_timeout: DEFAULT_RTS_TIMEOUT,
            control_retries: DEFAULT_CONTROL_RETRIES,
        }
    }
}

impl PbProbeConfig {
    pub fn udp_payload_bytes(&self) -> usize {
        let overhead = usize::from(OUTER_IPV6_HEADER_BYTES + OUTER_UDP_HEADER_BYTES);
        self.ip_packet_bytes.saturating_sub(overhead)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedSample {
    pub sample_id: u32,
    pub bulk_len: u32,
    pub delay_first: Duration,
    pub delay_last: Duration,
    pub dispersion: Duration,
    pub server_issue_duration: Option<Duration>,
}

impl AcceptedSample {
    pub fn delay_sum(self) -> Duration {
        self.delay_first.saturating_add(self.delay_last)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedSample {
    pub sample: AcceptedSample,
    pub capacity_mbps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    pub bulk_len: u32,
    pub sample_count: u32,
    pub attempts: u32,
    pub lost_samples: u32,
    pub ip_packet_bytes: usize,
    pub selected: SelectedSample,
    pub min_dispersion: Duration,
    pub server_issue_samples: u32,
    pub min_server_issue_duration: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EstimateOutcome {
    Complete(Estimate),
    IncreaseBulk {
        previous_bulk_len: u32,
        next_bulk_len: u32,
        observed_dispersion: Duration,
    },
}

pub fn select_capacity_sample(
    samples: &[AcceptedSample],
    ip_packet_bytes: usize,
) -> Option<SelectedSample> {
    let sample = samples
        .iter()
        .copied()
        .min_by_key(|sample| sample.delay_sum())?;
    let capacity_mbps = capacity_mbps(sample.bulk_len, ip_packet_bytes, sample.dispersion)?;
    Some(SelectedSample {
        sample,
        capacity_mbps,
    })
}

pub fn capacity_mbps(bulk_len: u32, ip_packet_bytes: usize, dispersion: Duration) -> Option<f64> {
    let nanos = dispersion.as_nanos();
    if bulk_len == 0 || ip_packet_bytes == 0 || nanos == 0 {
        return None;
    }

    let bits = f64::from(bulk_len) * (ip_packet_bytes as f64) * 8.0;
    Some(bits * 1_000.0 / (nanos as f64))
}

pub fn next_bulk_len(current: u32, max: u32) -> Option<u32> {
    let next = current.checked_mul(10)?;
    if next > max || next == current {
        return None;
    }
    Some(next)
}

pub fn pacing_interval(dispersion: Duration, utilization: f64) -> Option<Duration> {
    if dispersion.is_zero() || !utilization.is_finite() || utilization <= 0.0 {
        return None;
    }

    Some(Duration::from_secs_f64(
        (2.0 * dispersion.as_secs_f64()) / utilization,
    ))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        AcceptedSample, capacity_mbps, next_bulk_len, pacing_interval, select_capacity_sample,
    };

    #[test]
    fn capacity_uses_bulk_length_not_packet_count() {
        let estimate = capacity_mbps(100, 1500, Duration::from_micros(1200));
        assert_eq!(estimate, Some(1000.0));
    }

    #[test]
    fn selector_uses_minimum_delay_sum() {
        let samples = [
            AcceptedSample {
                sample_id: 1,
                bulk_len: 10,
                delay_first: Duration::from_millis(3),
                delay_last: Duration::from_millis(4),
                dispersion: Duration::from_micros(900),
                server_issue_duration: None,
            },
            AcceptedSample {
                sample_id: 2,
                bulk_len: 10,
                delay_first: Duration::from_millis(1),
                delay_last: Duration::from_millis(2),
                dispersion: Duration::from_micros(1200),
                server_issue_duration: None,
            },
        ];

        let selected = select_capacity_sample(&samples, 1500).expect("sample should be selected");
        assert_eq!(selected.sample.sample_id, 2);
        assert_eq!(selected.capacity_mbps, 100.0);
    }

    #[test]
    fn bulk_growth_is_tenfold_and_capped() {
        assert_eq!(next_bulk_len(1, 1000), Some(10));
        assert_eq!(next_bulk_len(1000, 1000), None);
    }

    #[test]
    fn pacing_follows_paper_formula() {
        let interval = pacing_interval(Duration::from_millis(1), 0.01);
        assert_eq!(interval, Some(Duration::from_millis(200)));
    }
}
