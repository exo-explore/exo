use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    pub sent: u32,
    pub received: u32,
    pub loss_ratio: f64,
    pub min: Duration,
    pub avg: Duration,
    pub max: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacitySample {
    pub sent_packets: u32,
    pub received_packets: u32,
    pub received_bytes: u64,
    pub span: Duration,
}

impl CapacitySample {
    pub fn loss_ratio(self) -> f64 {
        if self.sent_packets == 0 {
            return 0.0;
        }
        let lost = self.sent_packets.saturating_sub(self.received_packets);
        f64::from(lost) / f64::from(self.sent_packets)
    }

    pub fn mbps(self) -> Option<f64> {
        capacity_mbps(self.received_bytes, self.span)
    }
}

pub fn latency_stats(sent: u32, samples: &[Duration]) -> Option<LatencyStats> {
    let received = u32::try_from(samples.len()).ok()?;
    if sent == 0 || samples.is_empty() {
        return None;
    }

    let mut min = samples.first().copied()?;
    let mut max = min;
    let mut total_nanos = 0_u128;

    for sample in samples {
        min = min.min(*sample);
        max = max.max(*sample);
        total_nanos = total_nanos.saturating_add(sample.as_nanos());
    }

    let avg_nanos = total_nanos / u128::from(received);
    let avg = Duration::from_nanos(u64_saturating_from_u128(avg_nanos));
    let lost = sent.saturating_sub(received);

    Some(LatencyStats {
        sent,
        received,
        loss_ratio: f64::from(lost) / f64::from(sent),
        min,
        avg,
        max,
    })
}

pub fn capacity_mbps(received_bytes: u64, span: Duration) -> Option<f64> {
    let nanos = span.as_nanos();
    if received_bytes == 0 || nanos == 0 {
        return None;
    }

    let bits = received_bytes.saturating_mul(8);
    Some((bits as f64) * 1_000.0 / (nanos as f64))
}

pub fn duration_nanos_u64(duration: Duration) -> u64 {
    u64_saturating_from_u128(duration.as_nanos())
}

fn u64_saturating_from_u128(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{CapacitySample, capacity_mbps, latency_stats};

    #[test]
    fn latency_summary_reports_loss_and_bounds() {
        let samples = [
            Duration::from_millis(3),
            Duration::from_millis(1),
            Duration::from_millis(2),
        ];

        let Some(stats) = latency_stats(4, &samples) else {
            panic!("expected latency stats");
        };

        assert_eq!(stats.sent, 4);
        assert_eq!(stats.received, 3);
        assert_eq!(stats.loss_ratio, 0.25);
        assert_eq!(stats.min, Duration::from_millis(1));
        assert_eq!(stats.avg, Duration::from_millis(2));
        assert_eq!(stats.max, Duration::from_millis(3));
    }

    #[test]
    fn capacity_summary_reports_mbps() {
        let sample = CapacitySample {
            sent_packets: 10,
            received_packets: 10,
            received_bytes: 125_000,
            span: Duration::from_millis(1),
        };

        assert_eq!(sample.loss_ratio(), 0.0);
        assert_eq!(
            capacity_mbps(sample.received_bytes, sample.span),
            Some(1000.0)
        );
    }
}
