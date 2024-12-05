from .device_capabilities import DeviceCapabilities
from typing import Dict, Set, Optional, NamedTuple
from dataclasses import dataclass

@dataclass
class PeerConnection:
  from_id: str
  to_id: str
  description: Optional[str] = None

  def __hash__(self):
    # Use both from_id and to_id for uniqueness in sets
    return hash((self.from_id, self.to_id))

  def __eq__(self, other):
    if not isinstance(other, PeerConnection):
      return False
    # Compare both from_id and to_id for equality
    return self.from_id == other.from_id and self.to_id == other.to_id

class Topology:
  def __init__(self):
    self.nodes: Dict[str, DeviceCapabilities] = {}
    # Store PeerConnection objects in the adjacency lists
    self.peer_graph: Dict[str, Set[PeerConnection]] = {}
    self.active_node_id: Optional[str] = None

  def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
    self.nodes[node_id] = device_capabilities

  def get_node(self, node_id: str) -> DeviceCapabilities:
    return self.nodes.get(node_id)

  def all_nodes(self):
    return self.nodes.items()

  def add_edge(self, node1_id: str, node2_id: str, description: Optional[str] = None):
    if node1_id not in self.peer_graph:
      self.peer_graph[node1_id] = set()
    if node2_id not in self.peer_graph:
      self.peer_graph[node2_id] = set()

    # Create bidirectional connections with the same description
    conn1 = PeerConnection(node1_id, node2_id, description)
    conn2 = PeerConnection(node2_id, node1_id, description)

    self.peer_graph[node1_id].add(conn1)
    self.peer_graph[node2_id].add(conn2)

  def get_neighbors(self, node_id: str) -> Set[str]:
    # Convert PeerConnection objects back to just destination IDs
    return {conn.to_id for conn in self.peer_graph.get(node_id, set())}

  def all_edges(self):
    edges = []
    for node_id, connections in self.peer_graph.items():
      for conn in connections:
        # Only include each edge once by checking if reverse already exists
        if not any(e[0] == conn.to_id and e[1] == conn.from_id for e in edges):
          edges.append((conn.from_id, conn.to_id, conn.description))
    return edges

  def merge(self, other: "Topology"):
    for node_id, capabilities in other.nodes.items():
      self.update_node(node_id, capabilities)
    for node_id, connections in other.peer_graph.items():
      for conn in connections:
        self.add_edge(conn.from_id, conn.to_id, conn.description)

  def __str__(self):
    nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
    edges_str = ", ".join(f"{node}: {[f'{c.to_id}({c.description})' for c in conns]}"
                         for node, conns in self.peer_graph.items())
    return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"
