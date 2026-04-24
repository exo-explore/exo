//! Link-local profiling support.
//!
//! This module is intentionally independent of the Babel control plane.  The
//! standalone example uses it to measure one physical link directly; the daemon
//! can later consume the same types and estimators when route scoring is wired
//! in.

pub mod estimator;
pub mod pbprobe;
pub mod protocol;
pub mod socket;
pub mod standalone;
pub mod types;

pub use estimator::{CapacitySample, LatencyStats, capacity_mbps, latency_stats};
pub use types::{DEFAULT_PROFILE_PORT, LinkKey, ProbeConfig};
