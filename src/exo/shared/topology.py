import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.topology import (
    Connection,
    Cycle,
    RDMAConnection,
    SocketConnection,
)


class TopologySnapshot(BaseModel):
    nodes: Sequence[NodeId]
    connections: Iterable[Connection]

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass
class Topology:
    _graph: rx.PyDiGraph[NodeId, SocketConnection | RDMAConnection] = field(
        init=False, default_factory=rx.PyDiGraph
    )
    _vertex_indices: dict[NodeId, int] = field(init=False, default_factory=dict)

    def to_snapshot(self) -> TopologySnapshot:
        return TopologySnapshot(
            nodes=list(self.list_nodes()), connections=self.list_connections()
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
        topology = cls()

        for node_id in snapshot.nodes:
            with contextlib.suppress(ValueError):
                topology.add_node(node_id)

        for conn in snapshot.connections:
            topology.add_connection(conn)

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

    def out_edges(self, node_id: NodeId) -> Iterable[Connection]:
        if node_id not in self._vertex_indices:
            return []
        return (
            Connection(source=self._graph[source], sink=self._graph[sink], edge=edge)
            for source, sink, edge in self._graph.out_edges(
                self._vertex_indices[node_id]
            )
        )

    def contains_node(self, node_id: NodeId) -> bool:
        return node_id in self._vertex_indices

    def add_connection(self, conn: Connection) -> None:
        source, sink, edge = conn.source, conn.sink, conn.edge
        del conn
        if edge in self.get_all_connections_between(source, sink):
            return

        if source not in self._vertex_indices:
            self.add_node(source)
        if sink not in self._vertex_indices:
            self.add_node(sink)

        src_id = self._vertex_indices[source]
        sink_id = self._vertex_indices[sink]

        _ = self._graph.add_edge(src_id, sink_id, edge)

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
    ) -> Iterable[Connection]:
        return (
            (
                Connection(
                    source=self._graph[src_id],
                    sink=self._graph[sink_id],
                    edge=connection,
                )
            )
            for src_id, sink_id, connection in self._graph.weighted_edge_list()
        )

    def remove_node(self, node_id: NodeId) -> None:
        if node_id not in self._vertex_indices:
            return

        rx_idx = self._vertex_indices[node_id]
        self._graph.remove_node(rx_idx)

        del self._vertex_indices[node_id]

    def replace_all_out_rdma_connections(
        self, source: NodeId, new_connections: Sequence[Connection]
    ) -> None:
        for conn_idx in self._graph.out_edge_indices(self._vertex_indices[source]):
            if isinstance(self._graph.get_edge_data_by_index(conn_idx), RDMAConnection):
                self._graph.remove_edge_from_index(conn_idx)
        for conn in new_connections:
            self.add_connection(conn)

    def remove_connection(self, conn: Connection) -> None:
        if (
            conn.source not in self._vertex_indices
            or conn.sink not in self._vertex_indices
        ):
            return
        for conn_idx in self._graph.edge_indices_from_endpoints(
            self._vertex_indices[conn.source], self._vertex_indices[conn.sink]
        ):
            if self._graph.get_edge_data_by_index(conn_idx) == conn.edge:
                self._graph.remove_edge_from_index(conn_idx)

    def get_cycles(self) -> list[Cycle]:
        """Get simple cycles in the graph, including singleton cycles"""

        cycle_idxs = rx.simple_cycles(self._graph)
        cycles: list[Cycle] = []
        for cycle_idx in cycle_idxs:
            cycle = Cycle(node_ids=[self._graph[idx] for idx in cycle_idx])
            cycles.append(cycle)
        for node_id in self.list_nodes():
            cycles.append(Cycle(node_ids=[node_id]))
        return cycles

    def get_cycles_tb(self) -> list[Cycle]:
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
        cycles: list[Cycle] = []
        for cycle_idx in cycle_idxs:
            cycle = Cycle(node_ids=[tb_graph[idx] for idx in cycle_idx])
            cycles.append(cycle)

        return cycles

    def get_subgraph_from_nodes(self, node_ids: list[NodeId]) -> "Topology":
        topology = Topology()
        for node_id in node_ids:
            topology.add_node(node_id)
        for connection in self.list_connections():
            if connection.source in node_ids and connection.sink in node_ids:
                topology.add_connection(connection)
        return topology

    def is_thunderbolt_cycle(self, cycle: Cycle) -> bool:
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
