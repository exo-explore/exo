from enum import Enum

from loguru import logger

from exo.shared.types.multiaddr import Multiaddr
from exo.utils.pydantic_ext import FrozenModel


class RDMAConnection(FrozenModel):
    source_rdma_iface: str
    sink_rdma_iface: str

    def is_thunderbolt(self) -> bool:
        logger.warning("duh")
        return True


# TODO
class LinkType(str, Enum):
    Thunderbolt = "Thunderbolt"
    Ethernet = "Ethernet"
    WiFi = "WiFi"


class SocketConnection(FrozenModel):
    sink_multiaddr: Multiaddr

    def __hash__(self):
        return hash(self.sink_multiaddr.ip_address)

    def is_thunderbolt(self) -> bool:
        return str(self.sink_multiaddr.ipv4_address).startswith("169.254")
