import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    InterfaceType,
    NodeNetworkInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.topology import (
    Connection,
    Cycle,
    RDMAConnection,
    SocketConnection,
)


class TopologySnapshot(BaseModel):
    nodes: Sequence[NodeId]
    connections: Mapping[
        NodeId, Mapping[NodeId, Sequence[SocketConnection | RDMAConnection]]
    ]

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass
class Topology:
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
                for edge in snapshot.connections[source][sink]:
                    topology.add_connection(
                        Connection(source=source, sink=sink, edge=edge)
                    )

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

    def get_rdma_cycles(self) -> list[Cycle]:
        rdma_edges = [
            (u, v, conn)
            for u, v, conn in self._graph.weighted_edge_list()
            if isinstance(conn, RDMAConnection)
        ]

        rdma_graph: rx.PyDiGraph[NodeId, SocketConnection | RDMAConnection] = (
            rx.PyDiGraph()
        )
        rdma_graph.add_nodes_from(self._graph.nodes())

        for u, v, conn in rdma_edges:
            rdma_graph.add_edge(u, v, conn)

        cycle_idxs = rx.simple_cycles(rdma_graph)
        cycles: list[Cycle] = []
        for cycle_idx in cycle_idxs:
            cycle = Cycle(node_ids=[rdma_graph[idx] for idx in cycle_idx])
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

    def is_rdma_cycle(self, cycle: Cycle) -> bool:
        node_idxs = [node for node in cycle]
        rx_idxs = [self._vertex_indices[idx] for idx in node_idxs]
        for rid in rx_idxs:
            for neighbor_rid in self._graph.neighbors(rid):
                if neighbor_rid not in rx_idxs:
                    continue
                has_rdma = False
                for edge in self._graph.get_all_edge_data(rid, neighbor_rid):
                    if isinstance(edge, RDMAConnection):
                        has_rdma = True
                        break
                if not has_rdma:
                    return False
        return True

    def get_thunderbolt_bridge_cycles(
        self,
        node_tb_bridge_status: Mapping[NodeId, ThunderboltBridgeStatus],
        node_network: Mapping[NodeId, NodeNetworkInfo],
    ) -> list[list[NodeId]]:
        """
        Find cycles in the Thunderbolt topology where all nodes have TB bridge enabled.
        Only returns cycles with >=2 nodes (2+ machines in a loop), as
        1 node doesn't cause the broadcast storm problem.
        """
        enabled_nodes = {
            node_id
            for node_id, status in node_tb_bridge_status.items()
            if status.enabled
        }

        if len(enabled_nodes) < 2:
            return []

        thunderbolt_ips = _get_ips_with_interface_type(
            enabled_nodes, node_network, "thunderbolt"
        )

        # Build subgraph with only TB bridge enabled nodes and thunderbolt connections
        graph: rx.PyDiGraph[NodeId, SocketConnection | RDMAConnection] = rx.PyDiGraph()
        node_to_idx: dict[NodeId, int] = {}

        for node_id in enabled_nodes:
            if node_id in self._vertex_indices:
                node_to_idx[node_id] = graph.add_node(node_id)

        for u, v, conn in self._graph.weighted_edge_list():
            source_id, sink_id = self._graph[u], self._graph[v]
            if source_id not in node_to_idx or sink_id not in node_to_idx:
                continue
            # Include connection if it's over a thunderbolt interface
            if (
                isinstance(conn, SocketConnection)
                and conn.sink_multiaddr.ip_address in thunderbolt_ips
            ):
                graph.add_edge(node_to_idx[source_id], node_to_idx[sink_id], conn)
            if isinstance(conn, RDMAConnection):
                graph.add_edge(node_to_idx[source_id], node_to_idx[sink_id], conn)

        return [
            [graph[idx] for idx in cycle]
            for cycle in rx.simple_cycles(graph)
            if len(cycle) >= 2
        ]


def _get_ips_with_interface_type(
    node_ids: set[NodeId],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    interface_type: InterfaceType,
) -> set[str]:
    """Get all IP addresses on interfaces of the specified type for the given nodes."""
    ips: set[str] = set()
    for node_id in node_ids:
        network_info = node_network.get(node_id, NodeNetworkInfo())
        for iface in network_info.interfaces:
            if iface.interface_type == interface_type:
                ips.add(iface.ip_address)
    return ips
