import asyncio
import time
import traceback
from typing import List, Dict, Callable, Tuple
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from exo.networking.manual.read_config import ReadManualConfig
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle

class ManualDiscovery(Discovery):
  def __init__(
    self,
    create_peer_handle: Callable[[str, str, DeviceCapabilities], PeerHandle],
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    discovery_config: str = "topology.yml",
    discovery_interval: int = 5,
    discovery_timeout: int = 30,
    update_interval: int = 15,
  ):
    self.discovery_config = discovery_config
    self.list_device = ReadManualConfig(discovery_config=self.discovery_config)
    device_capabilities = self.list_device.device_capabilities(self.list_device.whoami)
    self.create_peer_handle = create_peer_handle
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.update_interval = update_interval
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.discovery_task = None
    self.cleanup_task = None
    self._device_id = None
    self.update_task = None

  async def start(self):
    self.discovery_task = asyncio.create_task(self.task_discover_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

  async def task_discover_peers(self):
    while True:
      try:
        print("task_discover_peers")
        current_time = time.time()

        for device in self.list_device.config_devices:
          if f"{self.list_device.whoami}" == f"{device['server']}": continue
          print(f"Adresse: {device['address']} {device['port']}")
          peer_id = f"{device['id']}"
          peer_host = f"{device['address']}"
          peer_port = f"{device['port']}"

          if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
            new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", self.list_device.device_capabilities((str((f"{device['server']}")))))
            if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
            self.known_peers[peer_id] = (
              new_peer_handle,
              current_time,
              current_time,
            )

      except Exception as e:
        print(f"Error in discover peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def stop(self):
    if self.discovery_task:
      self.discovery_task.cancel()
    if self.cleanup_task:
      self.cleanup_task.cancel()
    if self.discovery_task or self.cleanup_task:
      await asyncio.gather(self.discovery_task, self.cleanup_task, return_exceptions=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2:
          print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _ in self.known_peers.values()]

  async def task_cleanup_peers(self):
    while True:
      try:
        current_time = time.time()
        peers_to_remove = [
          peer_handle.id() for peer_handle, connected_at, last_seen in self.known_peers.values()
          if (not await peer_handle.is_connected() and current_time - connected_at > self.discovery_timeout) or current_time - last_seen > self.discovery_timeout or not await peer_handle.health_check()
        ]
        if DEBUG_DISCOVERY >= 2: print("Peer statuses:", {peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, {connected_at=}, {last_seen=}, health_check={await peer_handle.health_check()}" for peer_handle, connected_at, last_seen in self.known_peers.values()})
        for peer_id in peers_to_remove:
          if peer_id in self.known_peers: del self.known_peers[peer_id]
          if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)