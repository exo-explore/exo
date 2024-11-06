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
  ):
    self.topology = NetworkTopology.from_path(network_config_path)
    self.network_config_path = network_config_path
    self.node_id = node_id
    self.create_peer_handle = create_peer_handle

    if node_id not in self.topology.peers:
      raise ValueError(
        f"Node ID {node_id} not found in network config file {network_config_path}. Please run with `node_id` set to one of the keys in the config file: {[k for k, _ in self.topology.peers]}"
      )

    self.listen_task = None

    self.known_peers: Dict[str, PeerHandle] = {}
    self.peers_in_network: Dict[str, PeerConfig] = self.topology.peers
    self.peers_in_network.pop(node_id)

  async def start(self) -> None:
    self.listen_task = asyncio.create_task(self.task_find_peers_from_config())

  async def stop(self) -> None:
    if self.listen_task:
      self.listen_task.cancel()

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    if DEBUG_DISCOVERY >= 2: print(f"Discovered peers: {[peer.id() for peer in self.known_peers.values()]}")
    return list(self.known_peers.values())

  async def task_find_peers_from_config(self):
    if DEBUG_DISCOVERY >= 2: print("Starting task to find peers from config...")
    while True:
      for peer_id, peer_config in self.peers_in_network.items():
        try:
          if DEBUG_DISCOVERY >= 2: print(f"Checking peer {peer_id=} at {peer_config.address}:{peer_config.port}")
          peer = self.known_peers.get(peer_id)
          if not peer:
            if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} not found in known peers. Adding.")
            peer = self.create_peer_handle(peer_id, f"{peer_config.address}:{peer_config.port}", peer_config.device_capabilities)
          is_healthy = await peer.health_check()
          if is_healthy:
            if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is healthy.")
            self.known_peers[peer_id] = peer
          else:
            if DEBUG_DISCOVERY >= 2: print(f"{peer_id=} at {peer_config.address}:{peer_config.port} is not healthy.")
            try:
              del self.known_peers[peer_id]
            except KeyError:
              pass
        except Exception as e:
          if DEBUG_DISCOVERY >= 2: print(f"Exception occured when attempting to add {peer_id=}: {e}")
      await asyncio.sleep(1.0)

      if DEBUG_DISCOVERY >= 2: print(f"Current known peers: {[peer.id() for peer in self.known_peers.values()]}")

