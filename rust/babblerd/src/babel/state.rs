//! Reduced in-memory state derived from `babeld` event lines.
//!
//! This module is the consumer-side counterpart to [`crate::babel::line`]:
//!
//! - [`Event`] is the wire/domain event stream emitted by `babeld`
//! - [`BabelState`] is the current snapshot obtained by reducing those events
//!
//! The reducer model is intentionally simple:
//!
//! - `add` inserts the entity into the relevant table
//! - `change` upserts the entity into the relevant table
//! - `flush` removes the entity from the relevant table
//!
//! The stored state types do **not** retain [`EventKind`], because the event
//! kind is transport/update metadata rather than persistent object state.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};

use crate::babel::line::{
    Event, EventKind, InterfaceEvent, NeighbourEvent, RouteEvent, XRouteEvent,
};
use crate::babel::Eui64;
use ipnet::IpNet;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BabelState {
    pub interfaces: HashMap<Box<str>, InterfaceState>,
    pub neighbours: HashMap<u64, NeighbourState>,
    pub xroutes: HashMap<XRouteKey, XRouteState>,
    pub routes: HashMap<u64, RouteState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterfaceState {
    pub ifname: Box<str>,
    pub up: bool,
    pub ipv6: Option<IpAddr>,
    pub ipv4: Option<Ipv4Addr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbourState {
    pub handle: u64,
    pub address: IpAddr,
    pub ifname: Box<str>,
    pub reach: u16,
    pub ureach: u16,
    pub rxcost: u32,
    pub txcost: u32,
    pub rtt_millis: Option<u32>,
    pub rttcost: Option<u32>,
    pub cost: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct XRouteKey {
    pub prefix: IpNet,
    pub from: IpNet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XRouteState {
    pub prefix: IpNet,
    pub from: IpNet,
    pub metric: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteState {
    pub handle: u64,
    pub prefix: IpNet,
    pub from: IpNet,
    pub installed: bool,
    pub id: Eui64,
    pub metric: u32,
    pub refmetric: u32,
    pub via: IpAddr,
    pub ifname: Box<str>,
}

impl BabelState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, event: Event) {
        match event {
            Event::Interface(event) => self.apply_interface(event),
            Event::Neighbour(event) => self.apply_neighbour(event),
            Event::XRoute(event) => self.apply_xroute(event),
            Event::Route(event) => self.apply_route(event),
        }
    }

    pub fn extend<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = Event>,
    {
        for event in events {
            self.apply(event);
        }
    }

    fn apply_interface(&mut self, event: InterfaceEvent) {
        let key = event.ifname.clone();
        match event.kind {
            EventKind::Add | EventKind::Change => {
                self.interfaces.insert(key, event.into());
            }
            EventKind::Flush => {
                self.interfaces.remove(&key);
            }
        }
    }

    fn apply_neighbour(&mut self, event: NeighbourEvent) {
        let key = event.handle;
        match event.kind {
            EventKind::Add | EventKind::Change => {
                self.neighbours.insert(key, event.into());
            }
            EventKind::Flush => {
                self.neighbours.remove(&key);
            }
        }
    }

    fn apply_xroute(&mut self, event: XRouteEvent) {
        let key = XRouteKey {
            prefix: event.prefix,
            from: event.from,
        };
        match event.kind {
            EventKind::Add | EventKind::Change => {
                self.xroutes.insert(key, event.into());
            }
            EventKind::Flush => {
                self.xroutes.remove(&key);
            }
        }
    }

    fn apply_route(&mut self, event: RouteEvent) {
        let key = event.handle;
        match event.kind {
            EventKind::Add | EventKind::Change => {
                self.routes.insert(key, event.into());
            }
            EventKind::Flush => {
                self.routes.remove(&key);
            }
        }
    }
}

impl From<InterfaceEvent> for InterfaceState {
    fn from(event: InterfaceEvent) -> Self {
        Self {
            ifname: event.ifname,
            up: event.up,
            ipv6: event.ipv6,
            ipv4: event.ipv4,
        }
    }
}

impl From<NeighbourEvent> for NeighbourState {
    fn from(event: NeighbourEvent) -> Self {
        Self {
            handle: event.handle,
            address: event.address,
            ifname: event.ifname,
            reach: event.reach,
            ureach: event.ureach,
            rxcost: event.rxcost,
            txcost: event.txcost,
            rtt_millis: event.rtt_millis,
            rttcost: event.rttcost,
            cost: event.cost,
        }
    }
}

impl From<XRouteEvent> for XRouteState {
    fn from(event: XRouteEvent) -> Self {
        Self {
            prefix: event.prefix,
            from: event.from,
            metric: event.metric,
        }
    }
}

impl From<RouteEvent> for RouteState {
    fn from(event: RouteEvent) -> Self {
        Self {
            handle: event.handle,
            prefix: event.prefix,
            from: event.from,
            installed: event.installed,
            id: event.id,
            metric: event.metric,
            refmetric: event.refmetric,
            via: event.via,
            ifname: event.ifname,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use super::{BabelState, InterfaceState, XRouteKey};
    use crate::babel::line::{
        Event, EventKind, InterfaceEvent, NeighbourEvent, RouteEvent, XRouteEvent,
    };
    use crate::babel::Eui64;
    use ipnet::IpNet;

    fn net(s: &str) -> IpNet {
        s.parse().unwrap()
    }

    #[test]
    fn interface_add_change_flush() {
        let mut state = BabelState::new();

        state.apply(Event::Interface(InterfaceEvent {
            kind: EventKind::Add,
            ifname: "en2".into(),
            up: true,
            ipv6: Some(IpAddr::V6(Ipv6Addr::LOCALHOST)),
            ipv4: Some(Ipv4Addr::new(169, 254, 1, 2)),
        }));
        assert_eq!(
            state.interfaces.get("en2"),
            Some(&InterfaceState {
                ifname: "en2".into(),
                up: true,
                ipv6: Some(IpAddr::V6(Ipv6Addr::LOCALHOST)),
                ipv4: Some(Ipv4Addr::new(169, 254, 1, 2)),
            })
        );

        state.apply(Event::Interface(InterfaceEvent {
            kind: EventKind::Change,
            ifname: "en2".into(),
            up: false,
            ipv6: None,
            ipv4: None,
        }));
        assert_eq!(
            state.interfaces.get("en2"),
            Some(&InterfaceState {
                ifname: "en2".into(),
                up: false,
                ipv6: None,
                ipv4: None,
            })
        );

        state.apply(Event::Interface(InterfaceEvent {
            kind: EventKind::Flush,
            ifname: "en2".into(),
            up: false,
            ipv6: None,
            ipv4: None,
        }));
        assert!(!state.interfaces.contains_key("en2"));
    }

    #[test]
    fn neighbour_add_and_flush() {
        let mut state = BabelState::new();

        state.apply(Event::Neighbour(NeighbourEvent {
            kind: EventKind::Add,
            handle: 0xabc,
            address: IpAddr::V6("fe80::1".parse().unwrap()),
            ifname: "en3".into(),
            reach: 0x00ff,
            ureach: 0x000f,
            rxcost: 96,
            txcost: 128,
            rtt_millis: Some(42),
            rttcost: Some(10),
            cost: 224,
        }));
        assert_eq!(state.neighbours.len(), 1);
        assert_eq!(state.neighbours.get(&0xabc).unwrap().ifname.as_ref(), "en3");

        state.apply(Event::Neighbour(NeighbourEvent {
            kind: EventKind::Flush,
            handle: 0xabc,
            address: IpAddr::V6("fe80::1".parse().unwrap()),
            ifname: "en3".into(),
            reach: 0,
            ureach: 0,
            rxcost: 0,
            txcost: 0,
            rtt_millis: None,
            rttcost: None,
            cost: 0,
        }));
        assert!(state.neighbours.is_empty());
    }

    #[test]
    fn xroute_change_upserts_by_prefix_pair() {
        let mut state = BabelState::new();

        state.apply(Event::XRoute(XRouteEvent {
            kind: EventKind::Add,
            prefix: net("fde0:20c6:1fa7:ffff::/128"),
            from: net("::/0"),
            metric: 256,
        }));
        state.apply(Event::XRoute(XRouteEvent {
            kind: EventKind::Change,
            prefix: net("fde0:20c6:1fa7:ffff::/128"),
            from: net("::/0"),
            metric: 42,
        }));

        assert_eq!(state.xroutes.len(), 1);
        assert_eq!(
            state
                .xroutes
                .get(&XRouteKey {
                    prefix: net("fde0:20c6:1fa7:ffff::/128"),
                    from: net("::/0"),
                })
                .unwrap()
                .metric,
            42
        );
    }

    #[test]
    fn route_add_and_flush_by_handle() {
        let mut state = BabelState::new();

        state.apply(Event::Route(RouteEvent {
            kind: EventKind::Add,
            handle: 0xdeadbeef,
            prefix: net("fde0:20c6:1fa7:ffff::/128"),
            from: net("::/0"),
            installed: true,
            id: Eui64::new(0, 1, 2, 3, 4, 5, 6, 7),
            metric: 96,
            refmetric: 96,
            via: IpAddr::V6("fe80::1234".parse().unwrap()),
            ifname: "en2".into(),
        }));
        assert_eq!(state.routes.len(), 1);
        assert!(state.routes.get(&0xdeadbeef).unwrap().installed);

        state.apply(Event::Route(RouteEvent {
            kind: EventKind::Flush,
            handle: 0xdeadbeef,
            prefix: net("fde0:20c6:1fa7:ffff::/128"),
            from: net("::/0"),
            installed: false,
            id: Eui64::new(0, 1, 2, 3, 4, 5, 6, 7),
            metric: 0,
            refmetric: 0,
            via: IpAddr::V6("fe80::1234".parse().unwrap()),
            ifname: "en2".into(),
        }));
        assert!(state.routes.is_empty());
    }

    #[test]
    fn extend_applies_multiple_events() {
        let mut state = BabelState::new();
        state.extend([
            Event::Interface(InterfaceEvent {
                kind: EventKind::Add,
                ifname: "en2".into(),
                up: true,
                ipv6: None,
                ipv4: None,
            }),
            Event::XRoute(XRouteEvent {
                kind: EventKind::Add,
                prefix: net("fde0:20c6:1fa7:ffff::/128"),
                from: net("::/0"),
                metric: 123,
            }),
        ]);

        assert_eq!(state.interfaces.len(), 1);
        assert_eq!(state.xroutes.len(), 1);
    }
}
