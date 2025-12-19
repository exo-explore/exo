from loguru import logger

from exo.shared.types.multiaddr import Multiaddr
from exo.utils.pydantic_ext import CamelCaseModel

class TBConnection(CamelCaseModel):
    source_rdma_iface: str
    sink_rdma_iface: str

    def is_thunderbolt(self) -> bool:
        logger.warning("duh")
        return True

class Connection(CamelCaseModel):
    sink_multiaddr: Multiaddr

    def is_thunderbolt(self) -> bool:
        return str(self.sink_multiaddr.ipv4_address).startswith("169.254")
