from typing import Dict, Optional, Set
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.device_capabilities import DeviceCapabilities


class Topology:
  def __init__(self):
    self.nodes: Dict[str, DeviceCapabilities] = {}  # Maps node IDs to DeviceCapabilities
    self.peer_graph: Dict[str, Set[str]] = {}  # Adjacency list representing the graph
    self.active_node_id: Optional[str] = None
    self.file_ownership: Dict[str, Set[str]] = {}  # Maps file paths to node IDs

  def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
    self.nodes[node_id] = device_capabilities

  def get_node(self, node_id: str) -> DeviceCapabilities:
    return self.nodes.get(node_id)

  def all_nodes(self):
    return self.nodes.items()

  def add_edge(self, node1_id: str, node2_id: str):
    if node1_id not in self.peer_graph:
      self.peer_graph[node1_id] = set()
    if node2_id not in self.peer_graph:
      self.peer_graph[node2_id] = set()
    self.peer_graph[node1_id].add(node2_id)
    self.peer_graph[node2_id].add(node1_id)

  def get_neighbors(self, node_id: str) -> Set[str]:
    return self.peer_graph.get(node_id, set())

  def all_edges(self):
    edges = []
    for node, neighbors in self.peer_graph.items():
      for neighbor in neighbors:
        if (neighbor, node) not in edges:  # Avoid duplicate edges
          edges.append((node, neighbor))
    return edges

  def merge(self, other: "Topology"):
    for node_id, capabilities in other.nodes.items():
      self.update_node(node_id, capabilities)
    for node_id, neighbors in other.peer_graph.items():
      for neighbor in neighbors:
        self.add_edge(node_id, neighbor)

  def __str__(self):
    nodes_str = ", ".join(f"{node_id}: {cap}" for node_id, cap in self.nodes.items())
    edges_str = ", ".join(f"{node}: {neighbors}" for node, neighbors in self.peer_graph.items())
    return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"

  def update_file_ownership(self, node_id: str, file_path: str):
    """
    Updates the file ownership dictionary with the given node ID and file path.
    If the file path does not exist, it is added.
    """
    if file_path not in self.file_ownership:
      self.file_ownership[file_path] = set()
    self.file_ownership[file_path].add(node_id)

  async def send_broadcast_request(self, node_id: str, file_path: str) -> Optional[str]:
    """
    Sends a file request to the specified node using gRPC.
    If the node has the file, returns the node ID. Otherwise, returns None.
    """
    peer_handle = GRPCPeerHandle(node_id)
    has_file = await peer_handle.check_file(file_path)
    if has_file:
      self.update_file_ownership(node_id, file_path)
      return node_id
    return None

  async def download_from_peer(self, file_path: str, save_directory: str):
    """
    Initiates a file download from a peer that has the file.
    If no peer has the file, raises a ValueError.
    """
    nodes_with_file = await self.broadcast_file_request(file_path)
    if not nodes_with_file:
      raise ValueError(f"No peer has the file {file_path}")
    # Choose the first node with the file to download from
    node_id = nodes_with_file[0]
    peer_handle = GRPCPeerHandle(node_id)
    await peer_handle.download_file(file_path, save_directory)
