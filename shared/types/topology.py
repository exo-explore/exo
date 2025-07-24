from typing import Iterable, Protocol

from pydantic import BaseModel, ConfigDict

from shared.types.common import NodeId
from shared.types.profiling import ConnectionProfile, NodePerformanceProfile


class Connection(BaseModel):
    source_node_id: NodeId
    sink_node_id: NodeId
    source_multiaddr: str
    sink_multiaddr: str
    connection_profile: ConnectionProfile | None = None

    # required for Connection to be used as a key
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    def __hash__(self) -> int:
            return hash(
                (
                    self.source_node_id,
                    self.sink_node_id,
                    self.source_multiaddr,
                    self.sink_multiaddr,
                )
            )
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            raise ValueError("Cannot compare Connection with non-Connection")
        return (
            self.source_node_id == other.source_node_id
            and self.sink_node_id == other.sink_node_id
            and self.source_multiaddr == other.source_multiaddr
            and self.sink_multiaddr == other.sink_multiaddr
        )


class Node(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile | None = None


class TopologyProto(Protocol):
    def add_node(self, node: Node, node_id: NodeId) -> None: ...

    def add_connection(
        self,
        connection: Connection,
    ) -> None: ...

    def list_nodes(self) -> Iterable[Node]: ...

    def list_connections(self) -> Iterable[Connection]: ...

    def update_node_profile(self, node_id: NodeId, node_profile: NodePerformanceProfile) -> None: ...

    def update_connection_profile(self, connection: Connection) -> None: ...

    def remove_connection(self, connection: Connection) -> None: ...

    def remove_node(self, node_id: NodeId) -> None: ...

    def get_node_profile(self, node_id: NodeId) -> NodePerformanceProfile | None: ...

    def get_connection_profile(self, connection: Connection) -> ConnectionProfile | None: ...

    def get_cycles(self) -> list[list[Node]]: ...
