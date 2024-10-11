from .device_capabilities import DeviceCapabilities
from typing import Dict, Set, Optional


class Topology:
    def __init__(self):
        self.nodes: Dict[str, DeviceCapabilities] = {}
        self.peer_graph: Dict[str, Dict[str, float]] = {}
        self.active_node_id: Optional[str] = None

    def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
        self.nodes[node_id] = device_capabilities

    def get_node(self, node_id: str) -> DeviceCapabilities:
        return self.nodes.get(node_id)

    def all_nodes(self):
        return self.nodes.items()

    def add_edge(self, node1_id: str, node2_id: str, latency: float):
        if node1_id not in self.peer_graph:
            self.peer_graph[node1_id] = {}
        if node2_id not in self.peer_graph:
            self.peer_graph[node2_id] = {}
        self.peer_graph[node1_id][node2_id] = latency
        self.peer_graph[node2_id][node1_id] = latency

    def get_neighbors(self, node_id: str) -> Dict[str, float]:
        return self.peer_graph.get(node_id, {})

    def all_edges(self):
        edges = []
        for node, neighbors in self.peer_graph.items():
            for neighbor, latency in neighbors.items():
                if (neighbor, node, latency) not in edges:
                    edges.append((node, neighbor, latency))
        return edges

    def merge(self, other: "Topology"):
        for node_id, capabilities in other.nodes.items():
            self.update_node(node_id, capabilities)
        for node_id, neighbors in other.peer_graph.items():
            for neighbor_id, latency in neighbors.items():
                self.add_edge(node_id, neighbor_id, latency)

    def __str__(self):
        nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
        edges_str = ", ".join(f"{node}: {neighbors}" for node, neighbors in self.peer_graph.items())
        return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"