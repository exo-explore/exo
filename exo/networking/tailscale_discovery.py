import asyncio
import time
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Callable, Tuple
from tailscale import Tailscale, Device
from .discovery import Discovery
from .peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from .tailscale_helpers import get_device_id, update_device_attributes, get_device_attributes, update_device_attributes

class TailscaleDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    create_peer_handle: Callable[[str, str, DeviceCapabilities], PeerHandle],
    discovery_interval: int = 10,
    discovery_timeout: int = 30,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    tailscale_api_key: str = None,
    tailnet: str = None
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.create_peer_handle = create_peer_handle
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.discovery_task = None
    self.cleanup_task = None
    self.tailscale = Tailscale(api_key=tailscale_api_key, tailnet=tailnet)
    self._device_id = None

  async def start(self):
    self.device_capabilities = device_capabilities()
    await self.update_device_posture_attributes()  # Fetch and update device posture attributes
    self.discovery_task = asyncio.create_task(self.task_discover_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

  async def get_device_id(self):
    if self._device_id:
      return self._device_id
    self._device_id = await get_device_id()
    return self._device_id

  async def update_device_posture_attributes(self):
    await update_device_attributes(await self.get_device_id(), self.tailscale.api_key, self.node_id, self.node_port, self.device_capabilities)

  async def task_discover_peers(self):
    while True:
      try:
        devices: dict[str, Device] = await self.tailscale.devices()
        current_time = datetime.now(timezone.utc).timestamp()

        active_devices = {
          name: device for name, device in devices.items()
          if device.last_seen is not None and (current_time - device.last_seen.timestamp()) < 30
        }

        if DEBUG_DISCOVERY >= 4: print(f"Found tailscale devices: {devices}")
        if DEBUG_DISCOVERY >= 2: print(f"Active tailscale devices: {len(active_devices)}/{len(devices)}")
        if DEBUG_DISCOVERY >= 2: print("Time since last seen tailscale devices", [(current_time  - device.last_seen.timestamp()) for device in devices.values()])

        for device in active_devices.values():
          if device.name != self.node_id:
            peer_host = device.addresses[0]
            peer_id, peer_port, device_capabilities = await get_device_attributes(device.device_id, self.tailscale.api_key)
            print("retrieved attributes", peer_id, peer_host, peer_port, device_capabilities)

            if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
              if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
              self.known_peers[peer_id] = (
                self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", device_capabilities),
                current_time,
                current_time,
              )
            else:
              self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], current_time)

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
          if (not await peer_handle.is_connected() and current_time - connected_at > self.discovery_timeout) or current_time - last_seen > self.discovery_timeout
        ]
        if DEBUG_DISCOVERY >= 2:
          print("Peer statuses:", {peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, {connected_at=}, {last_seen=}" for peer_handle, connected_at, last_seen in self.known_peers.values()})
        for peer_id in peers_to_remove:
          if peer_id in self.known_peers:
            del self.known_peers[peer_id]
          if DEBUG_DISCOVERY >= 2:
            print(f"Removed peer {peer_id} due to inactivity.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)
