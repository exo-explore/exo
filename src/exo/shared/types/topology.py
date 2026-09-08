from collections.abc import Iterator
from dataclasses import dataclass

from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.utils.pydantic_ext import FrozenModel


@dataclass(frozen=True)
class Cycle:
    node_ids: list[NodeId]

    def __len__(self) -> int:
        return self.node_ids.__len__()

    def __iter__(self) -> Iterator[NodeId]:
        return self.node_ids.__iter__()


class RDMAConnection(FrozenModel):
    source_rdma_iface: str
    sink_rdma_iface: str


class SocketConnection(FrozenModel):
    sink_multiaddr: Multiaddr
    # Most recent reachability-probe round-trip time. Excluded from identity so
    # jitter does not make otherwise-identical edges unequal.
    latency_ms: float | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SocketConnection):
            return NotImplemented
        return (
            self.sink_multiaddr.ip_address == other.sink_multiaddr.ip_address
            and self.sink_multiaddr.port == other.sink_multiaddr.port
        )

    def __hash__(self):
        return hash(self.sink_multiaddr.ip_address)


class Connection(FrozenModel):
    source: NodeId
    sink: NodeId
    edge: RDMAConnection | SocketConnection
