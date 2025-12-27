import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.topology import RDMAConnection, SocketConnection


class TopologySnapshot(BaseModel):
    nodes: Sequence[NodeId]
    connections: Mapping[
        NodeId, Mapping[NodeId, Sequence[SocketConnection | RDMAConnection]]
    ]

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass
class Topology:
    # the _graph can be used as a int -> NodeId map.
    _graph: rx.PyDiGraph[NodeId, SocketConnection | RDMAConnection] = field(
        init=False, default_factory=rx.PyDiGraph
    )
    _vertex_indices: dict[NodeId, int] = field(init=False, default_factory=dict)

    def to_snapshot(self) -> TopologySnapshot:
        return TopologySnapshot(
            nodes=list(self.list_nodes()), connections=self.map_connections()
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
        topology = cls()

        for node_id in snapshot.nodes:
            with contextlib.suppress(ValueError):
                topology.add_node(node_id)

        for source in snapshot.connections:
            for sink in snapshot.connections[source]:
                for conn in snapshot.connections[source][sink]:
                    topology.add_connection(source, sink, conn)

        return topology

    def add_node(self, node_id: NodeId) -> None:
        if node_id in self._vertex_indices:
            return
        rx_id = self._graph.add_node(node_id)
        self._vertex_indices[node_id] = rx_id

    def node_is_leaf(self, node_id: NodeId) -> bool:
        return (
            node_id in self._vertex_indices
            and len(self._graph.neighbors(self._vertex_indices[node_id])) <= 1
        )

    def neighbours(self, node_id: NodeId) -> list[NodeId]:
        return [
            self._graph[rx_id]
            for rx_id in self._graph.neighbors(self._vertex_indices[node_id])
        ]

    def out_edges(
        self, node_id: NodeId
    ) -> Iterable[tuple[NodeId, SocketConnection | RDMAConnection]]:
        if node_id not in self._vertex_indices:
            return []
        return (
            (self._graph[nid], conn)
            for _, nid, conn in self._graph.out_edges(self._vertex_indices[node_id])
        )

    def contains_node(self, node_id: NodeId) -> bool:
        return node_id in self._vertex_indices

    def add_connection(
        self,
        source: NodeId,
        sink: NodeId,
        connection: SocketConnection | RDMAConnection,
    ) -> None:
        if connection in self.get_all_connections_between(source, sink):
            return

        if source not in self._vertex_indices:
            self.add_node(source)
        if sink not in self._vertex_indices:
            self.add_node(sink)

        src_id = self._vertex_indices[source]
        sink_id = self._vertex_indices[sink]

        _ = self._graph.add_edge(src_id, sink_id, connection)

    def get_all_connections_between(
        self, source: NodeId, sink: NodeId
    ) -> Iterable[SocketConnection | RDMAConnection]:
        if source not in self._vertex_indices:
            return []
        if sink not in self._vertex_indices:
            return []

        src_id = self._vertex_indices[source]
        sink_id = self._vertex_indices[sink]
        try:
            return self._graph.get_all_edge_data(src_id, sink_id)
        except rx.NoEdgeBetweenNodes:
            return []

    def list_nodes(self) -> Iterable[NodeId]:
        return self._graph.nodes()

    def map_connections(
        self,
    ) -> Mapping[NodeId, Mapping[NodeId, Sequence[SocketConnection | RDMAConnection]]]:
        base: dict[NodeId, dict[NodeId, list[SocketConnection | RDMAConnection]]] = {}
        for src_id, sink_id, connection in self._graph.weighted_edge_list():
            source = self._graph[src_id]
            sink = self._graph[sink_id]
            if source not in base:
                base[source] = {}
            if sink not in base[source]:
                base[source][sink] = []
            base[source][sink].append(connection)
        return base

    def list_connections(
        self,
    ) -> Iterable[tuple[NodeId, NodeId, SocketConnection | RDMAConnection]]:
        return (
            (
                self._graph[src_id],
                self._graph[sink_id],
                connection,
            )
            for src_id, sink_id, connection in self._graph.weighted_edge_list()
        )

    def remove_node(self, node_id: NodeId) -> None:
        if node_id not in self._vertex_indices:
            return

        rx_idx = self._vertex_indices[node_id]
        self._graph.remove_node(rx_idx)

        del self._vertex_indices[node_id]

    def replace_all_out_tb_connections(
        self, source: NodeId, new_connections: Sequence[tuple[NodeId, RDMAConnection]]
    ) -> None:
        for conn_idx in self._graph.out_edge_indices(self._vertex_indices[source]):
            if isinstance(self._graph.get_edge_data_by_index(conn_idx), RDMAConnection):
                self._graph.remove_edge_from_index(conn_idx)
        for sink, conn in new_connections:
            self.add_connection(source, sink, conn)

    def remove_connection(
        self, source: NodeId, sink: NodeId, edge: SocketConnection | RDMAConnection
    ) -> None:
        if source not in self._vertex_indices or sink not in self._vertex_indices:
            return
        for conn_idx in self._graph.edge_indices_from_endpoints(
            self._vertex_indices[source], self._vertex_indices[sink]
        ):
            if self._graph.get_edge_data_by_index(conn_idx) == edge:
                self._graph.remove_edge_from_index(conn_idx)

    def get_cycles(self) -> list[list[NodeId]]:
        cycle_idxs = rx.simple_cycles(self._graph)
        cycles: list[list[NodeId]] = []
        for cycle_idx in cycle_idxs:
            cycle = [self._graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def get_cycles_tb(self) -> list[list[NodeId]]:
        tb_edges = [
            (u, v, conn)
            for u, v, conn in self._graph.weighted_edge_list()
            if conn.is_thunderbolt()
        ]

        tb_graph: rx.PyDiGraph[NodeId, SocketConnection] = rx.PyDiGraph()
        tb_graph.add_nodes_from(self._graph.nodes())

        for u, v, conn in tb_edges:
            if isinstance(conn, SocketConnection):
                tb_graph.add_edge(u, v, conn)

        cycle_idxs = rx.simple_cycles(tb_graph)
        cycles: list[list[NodeId]] = []
        for cycle_idx in cycle_idxs:
            cycle = [tb_graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def get_subgraph_from_nodes(self, node_ids: list[NodeId]) -> "Topology":
        rx_idxs = [self._vertex_indices[idx] for idx in node_ids]
        topology = Topology()
        for rx_idx in rx_idxs:
            topology.add_node(self._graph[rx_idx])
        for source, sink, connection in self.list_connections():
            if source in node_ids and sink in node_ids:
                topology.add_connection(source, sink, connection)
        return topology

    def is_thunderbolt_cycle(self, cycle: list[NodeId]) -> bool:
        node_idxs = [node for node in cycle]
        rx_idxs = [self._vertex_indices[idx] for idx in node_idxs]
        for rid in rx_idxs:
            for neighbor_rid in self._graph.neighbors(rid):
                if neighbor_rid not in rx_idxs:
                    continue
                has_tb = False
                for edge in self._graph.get_all_edge_data(rid, neighbor_rid):
                    if edge.is_thunderbolt():
                        has_tb = True
                        break
                if not has_tb:
                    return False
        return True
