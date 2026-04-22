//! Immutable forwarding snapshots derived from [`crate::babel::BabelState`].
//!
//! `BabelState` mirrors the control-plane view emitted by `babeld`.
//! `FibSnapshot` is the reduced dataplane view:
//!
//! - exact-match IPv6 host routes only for now,
//! - one immutable snapshot swapped wholesale into the dataplane,
//! - keyed for fast lookup rather than protocol fidelity.
//!
//! The v1 forwarding model is intentionally narrow:
//!
//! - local addresses are explicit inputs, not inferred from every xroute,
//! - only installed IPv6 `/128` routes are considered,
//! - only destination-based forwarding is modeled,
//! - routes with non-link-local next hops are ignored.

use std::net::Ipv6Addr;

use ahash::RandomState;
use hashbrown::{HashMap, HashSet, hash_map::Entry};
use ipnet::{IpNet, Ipv6Net};

use crate::babel::BabelState;
use crate::babel::state::RouteState;

pub type HostKey = u128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FibEntry {
    pub next_hop_ll: Ipv6Addr,
    pub ifname: Box<str>,
    pub mtu: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FibSnapshot {
    pub locals: HashSet<HostKey, RandomState>,
    pub routes: HashMap<HostKey, FibEntry, RandomState>,
}

#[derive(Debug, Clone)]
pub struct FibBuilder {
    local_addrs: Vec<Ipv6Addr>,
    route_mtu: u16,
}

impl FibBuilder {
    pub fn new<I>(local_addrs: I, route_mtu: u16) -> Self
    where
        I: IntoIterator<Item = Ipv6Addr>,
    {
        Self {
            local_addrs: local_addrs.into_iter().collect(),
            route_mtu,
        }
    }

    pub fn derive(&self, state: &BabelState) -> FibSnapshot {
        let mut locals =
            HashSet::with_capacity_and_hasher(self.local_addrs.len(), RandomState::new());
        for addr in &self.local_addrs {
            locals.insert(host_key(*addr));
        }

        let mut routes = HashMap::with_hasher(RandomState::new());
        let mut route_scores = HashMap::with_hasher(RandomState::new());

        let mut candidates: Vec<&RouteState> = state.routes.values().collect();
        candidates.sort_by(|left, right| {
            left.ifname
                .cmp(&right.ifname)
                .then_with(|| left.prefix.to_string().cmp(&right.prefix.to_string()))
                .then_with(|| left.metric.cmp(&right.metric))
                .then_with(|| left.refmetric.cmp(&right.refmetric))
                .then_with(|| left.handle.cmp(&right.handle))
        });

        for route in candidates {
            let Some((dst, next_hop_ll)) = route_to_host(route) else {
                continue;
            };
            if locals.contains(&dst) {
                continue;
            }

            let candidate = FibEntry {
                next_hop_ll,
                ifname: route.ifname.clone(),
                mtu: self.route_mtu,
            };
            let candidate_score = (route.metric, route.refmetric, route.handle);

            match routes.entry(dst) {
                Entry::Vacant(slot) => {
                    slot.insert(candidate);
                    route_scores.insert(dst, candidate_score);
                }
                Entry::Occupied(mut slot) => {
                    let Some(existing_score) = route_scores.get(&dst).copied() else {
                        slot.insert(candidate);
                        route_scores.insert(dst, candidate_score);
                        continue;
                    };

                    if candidate_score < existing_score {
                        slot.insert(candidate);
                        route_scores.insert(dst, candidate_score);
                    }
                }
            }
        }

        FibSnapshot { locals, routes }
    }
}

impl FibSnapshot {
    pub fn empty() -> Self {
        Self {
            locals: HashSet::with_hasher(RandomState::new()),
            routes: HashMap::with_hasher(RandomState::new()),
        }
    }

    pub fn from_node_addr(node_addr: Ipv6Net, state: &BabelState, route_mtu: u16) -> Self {
        FibBuilder::new([node_addr.addr()], route_mtu).derive(state)
    }

    pub fn is_local(&self, addr: Ipv6Addr) -> bool {
        self.locals.contains(&host_key(addr))
    }

    pub fn lookup(&self, addr: Ipv6Addr) -> Option<&FibEntry> {
        self.routes.get(&host_key(addr))
    }
}

pub fn host_key(addr: Ipv6Addr) -> HostKey {
    u128::from(addr)
}

fn route_to_host(route: &RouteState) -> Option<(HostKey, Ipv6Addr)> {
    if !route.installed {
        return None;
    }

    let IpNet::V6(prefix) = route.prefix else {
        return None;
    };
    if prefix.prefix_len() != 128 {
        return None;
    }

    let IpNet::V6(from) = route.from else {
        return None;
    };
    if from.prefix_len() != 0 || from.addr() != Ipv6Addr::UNSPECIFIED {
        return None;
    }

    let std::net::IpAddr::V6(via) = route.via else {
        return None;
    };
    if !via.is_unicast_link_local() {
        return None;
    }

    Some((host_key(prefix.addr()), via))
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv6Addr};

    use crate::babel::Eui64;
    use crate::babel::line::{Event, EventKind, RouteEvent};
    use crate::babel::state::BabelState;

    use super::FibBuilder;

    fn route(
        handle: u64,
        prefix: &str,
        from: &str,
        installed: bool,
        via: Ipv6Addr,
        ifname: &str,
        metric: u32,
        refmetric: u32,
    ) -> Event {
        Event::Route(RouteEvent {
            kind: EventKind::Add,
            handle,
            prefix: prefix.parse().unwrap(),
            from: from.parse().unwrap(),
            installed,
            id: Eui64::new(0, 1, 2, 3, 4, 5, 6, 7),
            metric,
            refmetric,
            via: IpAddr::V6(via),
            ifname: ifname.into(),
        })
    }

    #[test]
    fn derives_local_and_host_routes() {
        let mut state = BabelState::new();
        state.apply(route(
            1,
            "fde0::1234/128",
            "::/0",
            true,
            "fe80::1".parse().unwrap(),
            "en2",
            96,
            32,
        ));

        let fib = FibBuilder::new(["fde0::1".parse().unwrap()], 1452).derive(&state);

        assert!(fib.is_local("fde0::1".parse().unwrap()));
        let entry = fib.lookup("fde0::1234".parse().unwrap()).unwrap();
        assert_eq!(entry.next_hop_ll, "fe80::1".parse::<Ipv6Addr>().unwrap());
        assert_eq!(entry.ifname.as_ref(), "en2");
        assert_eq!(entry.mtu, 1452);
    }

    #[test]
    fn skips_non_installed_or_non_host_routes() {
        let mut state = BabelState::new();
        state.apply(route(
            1,
            "fde0::abcd/128",
            "::/0",
            false,
            "fe80::1".parse().unwrap(),
            "en2",
            96,
            32,
        ));
        state.apply(route(
            2,
            "fde0::/64",
            "::/0",
            true,
            "fe80::2".parse().unwrap(),
            "en3",
            96,
            32,
        ));

        let fib = FibBuilder::new(["fde0::1".parse().unwrap()], 1452).derive(&state);
        assert!(fib.routes.is_empty());
    }

    #[test]
    fn prefers_lower_metric_when_multiple_installed_routes_exist() {
        let mut state = BabelState::new();
        state.apply(route(
            1,
            "fde0::beef/128",
            "::/0",
            true,
            "fe80::1".parse().unwrap(),
            "en2",
            200,
            20,
        ));
        state.apply(route(
            2,
            "fde0::beef/128",
            "::/0",
            true,
            "fe80::2".parse().unwrap(),
            "en3",
            100,
            10,
        ));

        let fib = FibBuilder::new(["fde0::1".parse().unwrap()], 1452).derive(&state);
        let entry = fib.lookup("fde0::beef".parse().unwrap()).unwrap();
        assert_eq!(entry.next_hop_ll, "fe80::2".parse::<Ipv6Addr>().unwrap());
        assert_eq!(entry.ifname.as_ref(), "en3");
    }
}
