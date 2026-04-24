use std::io;
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6, UdpSocket};
use std::num::NonZeroU32;
use std::time::Duration;

use nix::net::if_::if_nametoindex;
use socket2::{Domain, Protocol, Socket, Type};

const PROFILE_SOCKET_BUFFER_BYTES: usize = 4 * 1024 * 1024;

pub fn open_link_local_udp(
    ifname: &str,
    port: u16,
    read_timeout: Option<Duration>,
) -> io::Result<(UdpSocket, u32)> {
    let ifindex = if_nametoindex(ifname).map_err(io::Error::from)?;
    let Some(nonzero_ifindex) = NonZeroU32::new(ifindex) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid ifindex for {ifname}"),
        ));
    };

    let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_only_v6(true)?;
    socket.set_recv_buffer_size(PROFILE_SOCKET_BUFFER_BYTES)?;
    socket.set_send_buffer_size(PROFILE_SOCKET_BUFFER_BYTES)?;

    #[cfg(target_os = "linux")]
    socket.bind_device(Some(ifname.as_bytes()))?;
    socket.bind_device_by_index_v6(Some(nonzero_ifindex))?;
    socket.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, port, 0, 0).into())?;

    let udp: UdpSocket = socket.into();
    udp.set_read_timeout(read_timeout)?;
    udp.set_write_timeout(read_timeout)?;
    Ok((udp, ifindex))
}

pub fn scoped_peer_addr(peer: Ipv6Addr, port: u16, ifindex: u32) -> SocketAddr {
    SocketAddr::V6(SocketAddrV6::new(peer, port, 0, ifindex))
}

pub fn with_default_scope(addr: SocketAddr, ifindex: u32) -> SocketAddr {
    match addr {
        SocketAddr::V6(v6) if v6.ip().is_unicast_link_local() && v6.scope_id() == 0 => {
            SocketAddr::V6(SocketAddrV6::new(
                *v6.ip(),
                v6.port(),
                v6.flowinfo(),
                ifindex,
            ))
        }
        other => other,
    }
}

pub fn parse_link_local_addr(raw: &str) -> Result<Ipv6Addr, std::net::AddrParseError> {
    let addr = raw.split_once('%').map_or(raw, |(addr, _scope)| addr);
    addr.parse()
}
