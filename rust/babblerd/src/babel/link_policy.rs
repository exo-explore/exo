//! Temporary link-selection policy for broad macOS interface admission.
//!
//! The current MVP admits every usable interface and then steers Babel by
//! assigning each `enN` neighbour an absolute synthetic base cost of `N * 100`,
//! except that `en0` and `en1` are assigned the largest finite Babel cost. This is
//! intentionally a stopgap until measured link scoring lands.

use std::net::IpAddr;

use crate::babel::command::{
    NEIGHBOUR_COST_BIAS_256_MAX, NeighbourCostBias256, NeighbourCostCoef256,
    NeighbourCostCommand,
};
use crate::babel::line::{EventKind, NeighbourEvent};
use crate::babel::state::NeighbourState;

const EN_INDEX_COST_UNITS: u64 = 100;
const FIXED_POINT_SCALE: u64 = 256;
const ABSOLUTE_COST_COEF_256: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DesiredNeighbourCost {
    bias_256: NeighbourCostBias256,
    coef_256: NeighbourCostCoef256,
}

pub(crate) fn command_for_neighbour_event(event: &NeighbourEvent) -> Option<NeighbourCostCommand> {
    if !matches!(event.kind, EventKind::Add | EventKind::Change) {
        return None;
    }

    command_for_neighbour(
        &event.ifname,
        event.address,
        event.external_bias_256,
        event.external_coef_256,
    )
}

pub(crate) fn command_for_neighbour_state(
    neighbour: &NeighbourState,
) -> Option<NeighbourCostCommand> {
    command_for_neighbour(
        &neighbour.ifname,
        neighbour.address,
        neighbour.external_bias_256,
        neighbour.external_coef_256,
    )
}

fn command_for_neighbour(
    ifname: &str,
    address: IpAddr,
    external_bias_256: i32,
    external_coef_256: u32,
) -> Option<NeighbourCostCommand> {
    let desired = desired_en_index_cost(ifname)?;
    if external_bias_256 == desired.bias_256.raw() && external_coef_256 == desired.coef_256.raw() {
        return None;
    }

    let IpAddr::V6(link_local_neighbour) = address else {
        return None;
    };

    NeighbourCostCommand::new(
        ifname,
        link_local_neighbour,
        desired.bias_256,
        desired.coef_256,
    )
    .ok()
}

fn desired_en_index_cost(ifname: &str) -> Option<DesiredNeighbourCost> {
    let index = parse_en_index(ifname)?;
    let bias_256 = if index <= 1 {
        NEIGHBOUR_COST_BIAS_256_MAX
    } else {
        let bias_256 = u64::from(index)
            .checked_mul(EN_INDEX_COST_UNITS)?
            .checked_mul(FIXED_POINT_SCALE)?;
        i32::try_from(bias_256).ok()?
    };
    let bias_256 = NeighbourCostBias256::new(bias_256)?;
    let coef_256 = NeighbourCostCoef256::new(ABSOLUTE_COST_COEF_256)
        .expect("absolute-cost coefficient is within babeld's accepted range");

    Some(DesiredNeighbourCost { bias_256, coef_256 })
}

fn parse_en_index(ifname: &str) -> Option<u32> {
    let suffix = ifname.strip_prefix("en")?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::{command_for_neighbour_event, command_for_neighbour_state};
    use crate::babel::line::{EventKind, NeighbourEvent};
    use crate::babel::state::NeighbourState;

    fn neighbour_event(ifname: &str, bias: i32, coef: u32) -> NeighbourEvent {
        NeighbourEvent {
            kind: EventKind::Change,
            handle: 0x42,
            address: "fe80::1".parse().unwrap(),
            ifname: ifname.into(),
            reach: 0xffff,
            ureach: 0,
            rxcost: 96,
            txcost: 96,
            rtt_millis: None,
            rttcost: None,
            external_bias_256: bias,
            external_coef_256: coef,
            cost: 96,
        }
    }

    fn neighbour_state(ifname: &str, bias: i32, coef: u32) -> NeighbourState {
        NeighbourState {
            handle: 0x42,
            address: "fe80::1".parse().unwrap(),
            ifname: ifname.into(),
            reach: 0xffff,
            ureach: 0,
            rxcost: 96,
            txcost: 96,
            rtt_millis: None,
            rttcost: None,
            external_bias_256: bias,
            external_coef_256: coef,
            cost: 96,
        }
    }

    #[test]
    fn en_index_policy_sets_absolute_cost() {
        let command = command_for_neighbour_event(&neighbour_event("en18", 0, 256)).unwrap();
        assert_eq!(
            command.to_string(),
            "neighbour-cost en18 fe80::1 bias-256 460800 coef-256 0"
        );
    }

    #[test]
    fn en_index_policy_deprioritizes_en0_and_en1() {
        let command = command_for_neighbour_event(&neighbour_event("en0", 0, 256)).unwrap();
        assert_eq!(
            command.to_string(),
            "neighbour-cost en0 fe80::1 bias-256 16776704 coef-256 0"
        );

        let command = command_for_neighbour_event(&neighbour_event("en1", 0, 256)).unwrap();
        assert_eq!(
            command.to_string(),
            "neighbour-cost en1 fe80::1 bias-256 16776704 coef-256 0"
        );
    }

    #[test]
    fn en_index_policy_skips_already_configured_neighbour() {
        assert!(command_for_neighbour_state(&neighbour_state("en2", 51_200, 0)).is_none());
        assert!(
            command_for_neighbour_state(&neighbour_state("en0", 16_776_704, 0)).is_none()
        );
        assert!(
            command_for_neighbour_state(&neighbour_state("en1", 16_776_704, 0)).is_none()
        );
    }

    #[test]
    fn en_index_policy_ignores_non_en_or_non_link_local_neighbours() {
        assert!(command_for_neighbour_event(&neighbour_event("awdl0", 0, 256)).is_none());

        let mut ipv4 = neighbour_event("en2", 0, 256);
        ipv4.address = IpAddr::V4(Ipv4Addr::new(169, 254, 1, 2));
        assert!(command_for_neighbour_event(&ipv4).is_none());

        let mut non_link_local = neighbour_event("en2", 0, 256);
        non_link_local.address = "2001:db8::1".parse().unwrap();
        assert!(command_for_neighbour_event(&non_link_local).is_none());
    }

    #[test]
    fn en_index_policy_ignores_flush_events() {
        let mut event = neighbour_event("en2", 0, 256);
        event.kind = EventKind::Flush;
        assert!(command_for_neighbour_event(&event).is_none());
    }
}
