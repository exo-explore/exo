from exo.routing.connection_message import IpAddress
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from exo.utils.pydantic_ext import CamelCaseModel


class NodeInfo(CamelCaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile | None = None


class Connection(CamelCaseModel):
    source_id: NodeId
    sink_id: NodeId
    sink_addr: IpAddress
    connection_profile: ConnectionProfile | None = None

    def __hash__(self) -> int:
        return hash(
            (
                self.source_id,
                self.sink_id,
                self.sink_addr,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            raise ValueError("Cannot compare Connection with non-Connection")
        return (
            self.source_id == other.source_id
            and self.sink_id == other.sink_id
            and self.sink_addr == other.sink_addr
        )

    def is_thunderbolt(self) -> bool:
        return str(self.sink_addr).startswith("169.254")
