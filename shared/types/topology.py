from typing import Iterable, Protocol

from pydantic import BaseModel, ConfigDict

from shared.types.common import NodeId
from shared.types.multiaddr import Multiaddr
from shared.types.profiling import ConnectionProfile, NodePerformanceProfile


class Connection(BaseModel):
    local_node_id: NodeId
    send_back_node_id: NodeId
    local_multiaddr: Multiaddr
    send_back_multiaddr: Multiaddr
    connection_profile: ConnectionProfile | None = None

    # required for Connection to be used as a key
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def __hash__(self) -> int:
        return hash(
            (
                self.local_node_id,
                self.send_back_node_id,
                self.local_multiaddr.ip_address,
                self.send_back_multiaddr.ip_address,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            raise ValueError("Cannot compare Connection with non-Connection")
        return (
                self.local_node_id == other.local_node_id
                and self.send_back_node_id == other.send_back_node_id
                and self.local_multiaddr.ip_address == other.local_multiaddr.ip_address
                and self.send_back_multiaddr.ip_address == other.send_back_multiaddr.ip_address
        )
        
    def is_thunderbolt(self) -> bool:
        return str(self.local_multiaddr.ip_address).startswith('169.254') and str(self.send_back_multiaddr.ip_address).startswith('169.254')


class Node(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile | None = None


class TopologyProto(Protocol):
    def add_node(self, node: Node) -> None: ...

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
