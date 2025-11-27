from ipaddress import IPv4Address, IPv6Address, ip_address

from pydantic import ConfigDict
from exo_pyo3_bindings import RustConnectionMessage

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import CamelCaseModel

"""Serialisable types for Connection Updates/Messages"""

IpAddress = IPv4Address | IPv6Address


class SocketAddress(CamelCaseModel):
    # could be the python IpAddress type if we're feeling fancy
    ip: IpAddress
    port: int
    zone_id: int | None

    model_config = ConfigDict(
        frozen=True,
    )





class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    ips: set[SocketAddress]

    @classmethod
    def from_rust(cls, message: RustConnectionMessage) -> "ConnectionMessage":
        return cls(
            node_id=NodeId(str(message.endpoint_id)),
            ips=set(
                # TODO: better handle fallible conversion
                SocketAddress(
                    ip=ip_address(addr.ip_addr()),
                    port=addr.port(),
                    zone_id=addr.zone_id(),
                )
                for addr in message.current_transport_addrs
            ),
        )
