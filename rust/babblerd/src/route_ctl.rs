use std::io;
use std::net::IpAddr;

use ipnet::Ipv6Net;
use nix::net::if_::if_nametoindex;
use route_manager::{Route, RouteManager};

use crate::Result;

fn route_destination(prefix: Ipv6Net) -> IpAddr {
    IpAddr::V6(prefix.trunc().addr())
}

fn is_overlay_route(route: &Route, prefix: Ipv6Net) -> bool {
    route.destination() == route_destination(prefix) && route.prefix() == prefix.prefix_len()
}

fn tun_if_index(tun_ifname: &str) -> io::Result<u32> {
    if_nametoindex(tun_ifname).map_err(io::Error::from)
}

pub fn ensure_overlay_route(prefix: Ipv6Net, tun_ifname: &str) -> Result<()> {
    let tun_ifindex = tun_if_index(tun_ifname)?;
    let desired =
        Route::new(route_destination(prefix), prefix.prefix_len()).with_if_index(tun_ifindex);
    let mut manager = RouteManager::new()?;
    let existing: Vec<Route> = manager
        .list()?
        .into_iter()
        .filter(|route| is_overlay_route(route, prefix))
        .collect();

    let already_present = existing
        .iter()
        .any(|route| route.if_index() == Some(tun_ifindex) && route.gateway().is_none());
    if already_present {
        return Ok(());
    }

    for route in existing {
        manager.delete(&route)?;
    }
    manager.add(&desired)?;
    Ok(())
}

pub fn remove_overlay_route(prefix: Ipv6Net) -> Result<()> {
    let mut manager = RouteManager::new()?;
    let existing: Vec<Route> = manager
        .list()?
        .into_iter()
        .filter(|route| is_overlay_route(route, prefix))
        .collect();

    for route in existing {
        manager.delete(&route)?;
    }
    Ok(())
}
