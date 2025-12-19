import contextlib
from collections.abc import Sequence
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.topology import RDMAConnection, SocketConnection


class TopologySnapshot(BaseModel):
    nodes: list[NodeId]
    connections: list[tuple[NodeId, NodeId, SocketConnection | RDMAConnection]]

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Topology:
    def __init__(self) -> None:
        self._graph: rx.PyDiGraph[NodeId, SocketConnection | RDMAConnection] = (
            rx.PyDiGraph()
        )
        self._node_id_to_rx_id_map: dict[NodeId, int] = dict()
        self._rx_id_to_node_id_map: dict[int, NodeId] = dict()
        self._edge_id_to_rx_id_map: dict[SocketConnection | RDMAConnection, int] = (
            dict()
        )

    def to_snapshot(self) -> TopologySnapshot:
        return TopologySnapshot(
            nodes=list(self.list_nodes()),
            connections=list(self.list_connections()),
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
        topology = cls()

        for node in snapshot.nodes:
            with contextlib.suppress(ValueError):
                topology.add_node(node)

        for source, sink, connection in snapshot.connections:
            topology.add_connection(source, sink, connection)

        return topology

    def add_node(self, node: NodeId) -> None:
        if node in self._node_id_to_rx_id_map:
            return
        rx_id = self._graph.add_node(node)
        self._node_id_to_rx_id_map[node] = rx_id
        self._rx_id_to_node_id_map[rx_id] = node

    def node_is_leaf(self, node_id: NodeId) -> bool:
        return (
            node_id in self._node_id_to_rx_id_map
            and len(self._graph.neighbors(self._node_id_to_rx_id_map[node_id])) == 1
        )

    def neighbours(self, node_id: NodeId) -> list[NodeId]:
        return [
            self._rx_id_to_node_id_map[rx_id]
            for rx_id in self._graph.neighbors(self._node_id_to_rx_id_map[node_id])
        ]

    def out_edges(
        self, node_id: NodeId
    ) -> list[tuple[NodeId, SocketConnection | RDMAConnection]]:
        if node_id not in self._node_id_to_rx_id_map:
            return []
        return [
            (self._rx_id_to_node_id_map[nid], conn)
            for _, nid, conn in self._graph.out_edges(
                self._node_id_to_rx_id_map[node_id]
            )
        ]

    def contains_node(self, node_id: NodeId) -> bool:
        return node_id in self._node_id_to_rx_id_map

    def contains_connection(
        self, connection: SocketConnection | RDMAConnection
    ) -> bool:
        return connection in self._edge_id_to_rx_id_map

    def add_connection(
        self,
        source: NodeId,
        sink: NodeId,
        connection: SocketConnection | RDMAConnection,
    ) -> None:
        if source not in self._node_id_to_rx_id_map:
            self.add_node(source)
        if sink not in self._node_id_to_rx_id_map:
            self.add_node(sink)

        if connection in self._edge_id_to_rx_id_map:
            return

        src_id = self._node_id_to_rx_id_map[source]
        sink_id = self._node_id_to_rx_id_map[sink]

        rx_id = self._graph.add_edge(src_id, sink_id, connection)
        self._edge_id_to_rx_id_map[connection] = rx_id

    def get_all_connections_between(
        self, source: NodeId, sink: NodeId
    ) -> Iterable[SocketConnection | RDMAConnection]:
        src_id = self._node_id_to_rx_id_map[source]
        sink_id = self._node_id_to_rx_id_map[sink]
        return self._graph.get_all_edge_data(src_id, sink_id)

    def list_nodes(self) -> Iterable[NodeId]:
        return (self._graph[i] for i in self._graph.node_indices())

    def list_connections(
        self,
    ) -> Iterable[tuple[NodeId, NodeId, SocketConnection | RDMAConnection]]:
        return (
            (
                self._rx_id_to_node_id_map[src_id],
                self._rx_id_to_node_id_map[sink_id],
                connection,
            )
            for src_id, sink_id, connection in self._graph.weighted_edge_list()
        )

    def remove_node(self, node_id: NodeId) -> None:
        if node_id not in self._node_id_to_rx_id_map:
            return

        for src, sink, connection in self.list_connections():
            if src == node_id or sink == node_id:
                self.remove_connection(connection)

        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph.remove_node(rx_idx)

        del self._node_id_to_rx_id_map[node_id]
        del self._rx_id_to_node_id_map[rx_idx]

    def replace_all_out_tb_connections(
        self, source: NodeId, new_connections: Sequence[tuple[NodeId, RDMAConnection]]
    ) -> None:
        for conn in self.out_edges(source):
            if isinstance(conn, RDMAConnection):
                self.remove_connection(conn)
        for sink, conn in new_connections:
            self.add_connection(source, sink, conn)

    def remove_connection(self, connection: SocketConnection | RDMAConnection) -> None:
        if connection not in self._edge_id_to_rx_id_map:
            return
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.remove_edge_from_index(rx_idx)
        del self._edge_id_to_rx_id_map[connection]

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

    def get_subgraph_from_nodes(self, nodes: list[NodeId]) -> "Topology":
        node_idxs = [node for node in nodes]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
        topology = Topology()
        for rx_idx in rx_idxs:
            topology.add_node(self._graph[rx_idx])
        for source, sink, connection in self.list_connections():
            if source in node_idxs and sink in node_idxs:
                topology.add_connection(source, sink, connection)
        return topology

    def is_thunderbolt_cycle(self, cycle: list[NodeId]) -> bool:
        node_idxs = [node for node in cycle]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
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
