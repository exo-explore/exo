from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from exo.utils.pydantic_ext import CamelCaseModel


class NodeInfo(CamelCaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile | None = None


class Connection(CamelCaseModel):
    local_node_id: NodeId
    send_back_node_id: NodeId
    send_back_multiaddr: Multiaddr | None
    connection_profile: ConnectionProfile | None = None

    def __hash__(self) -> int:
        if self.send_back_multiaddr:
            return hash(
                (
                    self.local_node_id,
                    self.send_back_node_id,
                    self.send_back_multiaddr.address,
                )
            )
        else:
            return hash((self.local_node_id, self.send_back_node_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            raise ValueError("Cannot compare Connection with non-Connection")
        return (
            self.local_node_id == other.local_node_id
            and self.send_back_node_id == other.send_back_node_id
            and self.send_back_multiaddr == other.send_back_multiaddr
        )

    def is_thunderbolt(self) -> bool:
        return self.send_back_multiaddr is not None and str(
            self.send_back_multiaddr.ipv4_address
        ).startswith("169.254")

    def reverse(self) -> "Connection":
        return Connection(
            local_node_id=self.send_back_node_id,
            send_back_node_id=self.local_node_id,
            send_back_multiaddr=None,
        )
