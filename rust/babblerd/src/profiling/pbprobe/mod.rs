//! Paper-faithful PBProbe implementation.
//!
//! PBProbe is a CapProbe-derived capacity estimator that uses packet bulks
//! instead of a single packet pair.  This module follows the paper algorithm
//! rather than the old C implementation's process/control structure.

pub mod estimator;
pub mod protocol;
pub mod standalone;

pub use estimator::{
    AcceptedSample, Estimate, EstimateOutcome, PbProbeConfig, SelectedSample, next_bulk_len,
    pacing_interval, select_capacity_sample,
};
