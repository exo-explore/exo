import contextlib
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from shared.types.common import NodeId
from shared.types.multiaddr import Multiaddr
from shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from shared.types.topology import Connection, Node, TopologyProto


class TopologySnapshot(BaseModel):
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
        return TopologySnapshot(
            nodes=list(self.list_nodes()),
            connections=list(self.list_connections()),
            master_node_id=self.master_node_id,
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
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
        
    def set_master_node_id(self, node_id: NodeId) -> None:
        self.master_node_id = node_id

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
        try:
            rx_idx = self._node_id_to_rx_id_map[node_id]
            return self._graph.get_node_data(rx_idx).node_profile
        except KeyError:
            return None
    
    def get_node_multiaddr(self, node_id: NodeId) -> Multiaddr:
        for connection in self.list_connections():
            if connection.local_node_id == node_id:
                return connection.local_multiaddr
            if connection.send_back_node_id == node_id:
                return connection.send_back_multiaddr
        raise ValueError(f"Node {node_id} is not connected to any other nodes")
    
    def update_node_profile(self, node_id: NodeId, node_profile: NodePerformanceProfile) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph[rx_idx].node_profile = node_profile

    def update_connection_profile(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.update_edge_by_index(rx_idx, connection)

    def get_connection_profile(self, connection: Connection) -> ConnectionProfile | None:
        try:
            rx_idx = self._edge_id_to_rx_id_map[connection]
            return self._graph.get_edge_data_by_index(rx_idx).connection_profile
        except KeyError:
            return None

    def remove_node(self, node_id: NodeId) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph.remove_node(rx_idx)

        del self._node_id_to_rx_id_map[node_id]
        del self._rx_id_to_node_id_map[rx_idx]

    def remove_connection(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        if self._is_bridge(connection):
            # Determine the reference node from which reachability is calculated.
            # Prefer a master node if the topology knows one; otherwise fall back to
            # the local end of the connection being removed.
            reference_node_id: NodeId = self.master_node_id if self.master_node_id is not None else connection.local_node_id
            orphan_node_ids = self._get_orphan_node_ids(reference_node_id, connection)
            for orphan_node_id in orphan_node_ids:
                orphan_node_rx_id = self._node_id_to_rx_id_map[orphan_node_id]
                self._graph.remove_node(orphan_node_rx_id)
                del self._node_id_to_rx_id_map[orphan_node_id]
                del self._rx_id_to_node_id_map[orphan_node_rx_id]
            
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
    
    def get_subgraph_from_nodes(self, nodes: list[Node]) -> "Topology":
        node_idxs = [node.node_id for node in nodes]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
        topology = Topology()
        for rx_idx in rx_idxs:
            topology.add_node(self._graph[rx_idx])
        for connection in self.list_connections():
            if connection.local_node_id in node_idxs and connection.send_back_node_id in node_idxs:
                topology.add_connection(connection)
        return topology

    def _is_bridge(self, connection: Connection) -> bool:
        """Check if removing this connection will orphan any nodes from the master."""
        if self.master_node_id is None:
            return False
        
        orphan_node_ids = self._get_orphan_node_ids(self.master_node_id, connection)
        return len(orphan_node_ids) > 0

    def _get_orphan_node_ids(self, master_node_id: NodeId, connection: Connection) -> list[NodeId]:
        """Return node_ids that become unreachable from `master_node_id` once `connection` is removed.

        A node is considered *orphaned* if there exists **no directed path** from
        the master node to that node after deleting the edge identified by
        ``connection``. This definition is strictly weaker than being in a
        different *strongly* connected component and more appropriate for
        directed networks where information only needs to flow *outwards* from
        the master.
        """
        edge_idx = self._edge_id_to_rx_id_map[connection]
        # Operate on a copy so the original topology remains intact while we
        # compute reachability.
        graph_copy: rx.PyDiGraph[Node, Connection] = self._graph.copy()
        graph_copy.remove_edge_from_index(edge_idx)

        if master_node_id not in self._node_id_to_rx_id_map:
            # If the provided master node isn't present we conservatively treat
            # every other node as orphaned.
            return list(self._node_id_to_rx_id_map.keys())

        master_rx_id = self._node_id_to_rx_id_map[master_node_id]

        # Nodes reachable by following outgoing edges from the master.
        reachable_rx_ids: set[int] = set(rx.descendants(graph_copy, master_rx_id))
        reachable_rx_ids.add(master_rx_id)

        # Every existing node index not reachable is orphaned.
        orphan_rx_ids = set(graph_copy.node_indices()) - reachable_rx_ids

        return [self._rx_id_to_node_id_map[rx_id] for rx_id in orphan_rx_ids if rx_id in self._rx_id_to_node_id_map]
