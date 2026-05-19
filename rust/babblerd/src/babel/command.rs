//! Typed representation of commands sent to `babeld`'s local socket.
//!
//! This is the outbound counterpart to [`crate::babel::line`]:
//!
//! - [`crate::babel::line`] models what `babeld` emits
//! - this module models the runtime control lines that `babblerd` sends
//!
//! The scope here is intentionally narrow: this module only models the local-socket
//! commands that `babblerd` currently issues at runtime.
//!
//! NOTE: spawn-time `-C` configuration strings are still assembled in the runtime layer for now.
//!       If you want to push the protocol model further, the next obvious extraction is a typed
//!       configuration/config-statement layer rather than more runtime socket commands.

use std::fmt;
use std::net::Ipv6Addr;

pub const BABEL_INFINITY: i32 = 65_535;
pub const NEIGHBOUR_COST_BIAS_256_MIN: i32 = -((BABEL_INFINITY - 1) * 256);
pub const NEIGHBOUR_COST_BIAS_256_MAX: i32 = (BABEL_INFINITY - 1) * 256;
pub const NEIGHBOUR_COST_COEF_256_MIN: u32 = 0;
pub const NEIGHBOUR_COST_COEF_256_MAX: u32 = 65_535;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BabelCommand {
    Dump,
    Monitor,
    Unmonitor,
    Quit,
    Interface(Box<str>),
    NeighbourCost(NeighbourCostCommand),
}

impl BabelCommand {
    /// Encode this command for the local `babeld` socket, including line framing.
    #[must_use]
    pub fn encode(&self) -> String {
        format!("{self}\n")
    }
}

/// Signed fixed-point additive neighbour-cost bias in units of 1/256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourCostBias256(i32);

impl NeighbourCostBias256 {
    #[must_use]
    pub fn new(value: i32) -> Option<Self> {
        (NEIGHBOUR_COST_BIAS_256_MIN..=NEIGHBOUR_COST_BIAS_256_MAX)
            .contains(&value)
            .then_some(Self(value))
    }

    #[must_use]
    pub const fn neutral() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn raw(self) -> i32 {
        self.0
    }
}

/// Unsigned fixed-point neighbour-cost multiplier in units of 1/256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourCostCoef256(u32);

impl NeighbourCostCoef256 {
    #[must_use]
    pub fn new(value: u32) -> Option<Self> {
        (NEIGHBOUR_COST_COEF_256_MIN..=NEIGHBOUR_COST_COEF_256_MAX)
            .contains(&value)
            .then_some(Self(value))
    }

    #[must_use]
    pub const fn neutral() -> Self {
        Self(256)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbourCostCommand {
    ifname: Box<str>,
    link_local_neighbour: Ipv6Addr,
    bias_256: NeighbourCostBias256,
    coef_256: NeighbourCostCoef256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighbourCostCommandError {
    NonLinkLocalNeighbour,
}

impl NeighbourCostCommand {
    pub fn new(
        ifname: impl Into<Box<str>>,
        link_local_neighbour: Ipv6Addr,
        bias_256: NeighbourCostBias256,
        coef_256: NeighbourCostCoef256,
    ) -> Result<Self, NeighbourCostCommandError> {
        if !link_local_neighbour.is_unicast_link_local() {
            return Err(NeighbourCostCommandError::NonLinkLocalNeighbour);
        }

        Ok(Self {
            ifname: ifname.into(),
            link_local_neighbour,
            bias_256,
            coef_256,
        })
    }

    pub fn neutral(
        ifname: impl Into<Box<str>>,
        link_local_neighbour: Ipv6Addr,
    ) -> Result<Self, NeighbourCostCommandError> {
        Self::new(
            ifname,
            link_local_neighbour,
            NeighbourCostBias256::neutral(),
            NeighbourCostCoef256::neutral(),
        )
    }
}

impl fmt::Display for BabelCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dump => f.write_str("dump"),
            Self::Monitor => f.write_str("monitor"),
            Self::Unmonitor => f.write_str("unmonitor"),
            Self::Quit => f.write_str("quit"),
            Self::Interface(ifname) => write!(f, "interface {ifname}"),
            Self::NeighbourCost(cmd) => write!(f, "{cmd}"),
        }
    }
}

impl fmt::Display for NeighbourCostCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "neighbour-cost {} {} bias-256 {} coef-256 {}",
            self.ifname,
            self.link_local_neighbour,
            self.bias_256.raw(),
            self.coef_256.raw()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;

    use super::{
        BabelCommand, NeighbourCostBias256, NeighbourCostCoef256, NeighbourCostCommand,
        NeighbourCostCommandError,
    };

    #[test]
    fn renders_commands() {
        assert_eq!(BabelCommand::Dump.to_string(), "dump");
        assert_eq!(BabelCommand::Monitor.to_string(), "monitor");
        assert_eq!(BabelCommand::Unmonitor.to_string(), "unmonitor");
        assert_eq!(BabelCommand::Quit.to_string(), "quit");
        assert_eq!(
            BabelCommand::Interface("en2".into()).to_string(),
            "interface en2"
        );
        assert_eq!(
            BabelCommand::NeighbourCost(
                NeighbourCostCommand::new(
                    "en18",
                    "fe80::42".parse().unwrap(),
                    NeighbourCostBias256::new(4_096).unwrap(),
                    NeighbourCostCoef256::new(128).unwrap(),
                )
                .unwrap()
            )
            .to_string(),
            "neighbour-cost en18 fe80::42 bias-256 4096 coef-256 128"
        );
    }

    #[test]
    fn encodes_commands() {
        assert_eq!(BabelCommand::Dump.encode(), "dump\n");
        assert_eq!(BabelCommand::Monitor.encode(), "monitor\n");
        assert_eq!(
            BabelCommand::Interface("en2".into()).encode(),
            "interface en2\n"
        );
        assert_eq!(
            BabelCommand::NeighbourCost(
                NeighbourCostCommand::neutral("en2", "fe80::1".parse().unwrap()).unwrap()
            )
            .encode(),
            "neighbour-cost en2 fe80::1 bias-256 0 coef-256 256\n"
        );
    }

    #[test]
    fn validates_neighbour_cost_fixed_point_ranges() {
        assert_eq!(
            NeighbourCostBias256::new(super::NEIGHBOUR_COST_BIAS_256_MIN)
                .unwrap()
                .raw(),
            -16_776_704
        );
        assert_eq!(
            NeighbourCostBias256::new(super::NEIGHBOUR_COST_BIAS_256_MAX)
                .unwrap()
                .raw(),
            16_776_704
        );
        assert!(NeighbourCostBias256::new(super::NEIGHBOUR_COST_BIAS_256_MIN - 1).is_none());
        assert!(NeighbourCostBias256::new(super::NEIGHBOUR_COST_BIAS_256_MAX + 1).is_none());

        assert_eq!(NeighbourCostCoef256::new(0).unwrap().raw(), 0);
        assert_eq!(
            NeighbourCostCoef256::new(super::NEIGHBOUR_COST_COEF_256_MAX)
                .unwrap()
                .raw(),
            65_535
        );
        assert!(NeighbourCostCoef256::new(super::NEIGHBOUR_COST_COEF_256_MAX + 1).is_none());
    }

    #[test]
    fn rejects_non_link_local_neighbour_cost_address() {
        assert_eq!(
            NeighbourCostCommand::neutral("en2", Ipv6Addr::LOCALHOST),
            Err(NeighbourCostCommandError::NonLinkLocalNeighbour)
        );
    }
}
