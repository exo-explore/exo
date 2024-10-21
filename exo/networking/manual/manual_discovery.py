import asyncio
from exo.networking.discovery import Discovery
from typing import Dict, List, Callable

from exo.topology.device_capabilities import DeviceCapabilities
from exo.networking.manual.network_topology_config import NetworkTopology, PeerConfig
from exo.helpers import DEBUG_DISCOVERY
from exo.networking.peer_handle import PeerHandle


class ManualDiscovery(Discovery):
  def __init__(
    self,
    network_config_path: str,
    node_id: str,
    create_peer_handle: Callable[[str, str, DeviceCapabilities], PeerHandle],
    discovery_timeout: int = 30,
  ):
    self.topology = NetworkTopology.from_path(network_config_path)
    self.node_id = node_id
    self.create_peer_handle = create_peer_handle
    self.discovery_timeout = discovery_timeout

    try:
      self.node = self.topology.peers[node_id]
    except KeyError as e:
      print(f"Node ID {node_id} not found in network config file {network_config_path}. Please run with `node_id` set to one of the keys in the config file: {[k for k, _ in self.topology.peers]}")
      raise e

    self.node_port = self.node.port

    self.listen_task = None
    self.cleanup_task = None

    self.known_peers: Dict[str, PeerHandle] = {}
    self.peers_in_network: Dict[str, PeerConfig] = self.topology.peers
    self.node_config = self.peers_in_network.pop(node_id)

  async def start(self) -> None:
    self.listen_task = asyncio.create_task(self.task_find_peers_from_config())

  async def stop(self) -> None:
    if self.listen_task:
      self.listen_task.cancel()

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if DEBUG_DISCOVERY >= 2: print("Starting discovery...")
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return list(self.known_peers.values())

  async def task_find_peers_from_config(self):
    if DEBUG_DISCOVERY >= 2: print("Starting task to find peers from config...")
    while True:
      for peer_id, peer_config in self.peers_in_network.items():
        if DEBUG_DISCOVERY >= 2: print(f"Checking peer {peer_id=} at {peer_config.address}:{peer_config.port}")
        peer = self.known_peers.get(peer_id)
        if not peer:
          if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} not found in known peers. Adding.")
          new_peer_handle = self.create_peer_handle(peer_id, f"{peer_config.address}:{peer_config.port}", peer_config.device_capabilities)
          self.known_peers[peer_id] = new_peer_handle
          peer = new_peer_handle

        is_healthy = await peer.health_check()
        if not is_healthy:
          if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is not healthy. Removing.")
          del self.known_peers[peer_id]
        elif DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is healthy.")

        if DEBUG_DISCOVERY >= 2: print(f"Current known peers: {[peer.id() for peer in self.known_peers.values()]}")

      await asyncio.sleep(self.discovery_timeout)
