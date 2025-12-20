from enum import Enum

from loguru import logger

from exo.shared.types.multiaddr import Multiaddr
from exo.utils.pydantic_ext import CamelCaseModel


class RDMAConnection(CamelCaseModel):
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


class SocketConnection(CamelCaseModel):
    sink_multiaddr: Multiaddr

    def is_thunderbolt(self) -> bool:
        return str(self.sink_multiaddr.ipv4_address).startswith("169.254")
