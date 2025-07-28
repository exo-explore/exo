import contextlib
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from shared.types.common import NodeId
from shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from shared.types.topology import Connection, Node, TopologyProto


class TopologySnapshot(BaseModel):
    """Immutable serialisable representation of a :class:`Topology`."""

    nodes: list[Node]
    connections: list[Connection]
    master_node_id: NodeId | None = None

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Topology(TopologyProto):
    def __init__(self) -> None:
        self._graph: rx.PyDiGraph[Node, Connection] = rx.PyDiGraph()
        self._node_id_to_rx_id_map: dict[NodeId, int] = dict()
        self._rx_id_to_node_id_map: dict[int, NodeId] = dict()
        self._edge_id_to_rx_id_map: dict[Connection, int] = dict()
        self.master_node_id: NodeId | None = None

    def to_snapshot(self) -> TopologySnapshot:
        """Return an immutable snapshot suitable for JSON serialisation."""

        return TopologySnapshot(
            nodes=list(self.list_nodes()),
            connections=list(self.list_connections()),
            master_node_id=self.master_node_id,
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
        """Reconstruct a :class:`Topology` from *snapshot*.

        The reconstructed topology is equivalent (w.r.t. nodes, connections
        and ``master_node_id``) to the original one that produced *snapshot*.
        """

        topology = cls()
        topology.master_node_id = snapshot.master_node_id

        for node in snapshot.nodes:
            with contextlib.suppress(ValueError):
                topology.add_node(node)

        for connection in snapshot.connections:
            topology.add_connection(connection)

        return topology

    def add_node(self, node: Node) -> None:
        if node.node_id in self._node_id_to_rx_id_map:
            return
        rx_id = self._graph.add_node(node)
        self._node_id_to_rx_id_map[node.node_id] = rx_id
        self._rx_id_to_node_id_map[rx_id] = node.node_id

    def contains_node(self, node_id: NodeId) -> bool:
        return node_id in self._node_id_to_rx_id_map

    def contains_connection(self, connection: Connection) -> bool:
        return connection in self._edge_id_to_rx_id_map

    def add_connection(
            self,
            connection: Connection,
    ) -> None:
        if connection.local_node_id not in self._node_id_to_rx_id_map:
            self.add_node(Node(node_id=connection.local_node_id))
        if connection.send_back_node_id not in self._node_id_to_rx_id_map:
            self.add_node(Node(node_id=connection.send_back_node_id))

        src_id = self._node_id_to_rx_id_map[connection.local_node_id]
        sink_id = self._node_id_to_rx_id_map[connection.send_back_node_id]

        rx_id = self._graph.add_edge(src_id, sink_id, connection)
        self._edge_id_to_rx_id_map[connection] = rx_id

    def list_nodes(self) -> Iterable[Node]:
        yield from (self._graph[i] for i in self._graph.node_indices())

    def list_connections(self) -> Iterable[Connection]:
        for (_, _, connection) in self._graph.weighted_edge_list():
            yield connection

    def get_node_profile(self, node_id: NodeId) -> NodePerformanceProfile | None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        return self._graph.get_node_data(rx_idx).node_profile

    def update_node_profile(self, node_id: NodeId, node_profile: NodePerformanceProfile) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph[rx_idx].node_profile = node_profile

    def update_connection_profile(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.update_edge_by_index(rx_idx, connection)

    def get_connection_profile(self, connection: Connection) -> ConnectionProfile | None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        return self._graph.get_edge_data_by_index(rx_idx).connection_profile

    def remove_node(self, node_id: NodeId) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph.remove_node(rx_idx)

        del self._node_id_to_rx_id_map[node_id]
        del self._rx_id_to_node_id_map[rx_idx]

    def remove_connection(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        if self._is_bridge(connection):
            orphan_node_ids = self._get_orphan_node_ids(connection.local_node_id, connection)
            for orphan_node_id in orphan_node_ids:
                orphan_node_rx_id = self._node_id_to_rx_id_map[orphan_node_id]
                self._graph.remove_node(orphan_node_rx_id)
                del self._node_id_to_rx_id_map[orphan_node_id]
                del self._rx_id_to_node_id_map[orphan_node_rx_id]
        else:
            self._graph.remove_edge_from_index(rx_idx)
            del self._edge_id_to_rx_id_map[connection]
            if rx_idx in self._rx_id_to_node_id_map:
                del self._rx_id_to_node_id_map[rx_idx]

    def get_cycles(self) -> list[list[Node]]:
        cycle_idxs = rx.simple_cycles(self._graph)
        cycles: list[list[Node]] = []
        for cycle_idx in cycle_idxs:
            cycle = [self._graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def _is_bridge(self, connection: Connection) -> bool:
        edge_idx = self._edge_id_to_rx_id_map[connection]
        graph_copy = self._graph.copy().to_undirected()
        components_before = rx.number_connected_components(graph_copy)

        graph_copy.remove_edge_from_index(edge_idx)
        components_after = rx.number_connected_components(graph_copy)

        return components_after > components_before

    def _get_orphan_node_ids(self, master_node_id: NodeId, connection: Connection) -> list[NodeId]:
        edge_idx = self._edge_id_to_rx_id_map[connection]
        graph_copy = self._graph.copy().to_undirected()
        graph_copy.remove_edge_from_index(edge_idx)
        components = rx.connected_components(graph_copy)

        orphan_node_rx_ids: set[int] = set()
        master_node_rx_id = self._node_id_to_rx_id_map[master_node_id]
        for component in components:
            if master_node_rx_id not in component:
                orphan_node_rx_ids.update(component)

        return [self._rx_id_to_node_id_map[rx_id] for rx_id in orphan_node_rx_ids]
