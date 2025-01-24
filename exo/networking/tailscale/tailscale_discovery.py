import asyncio
import time
import traceback
from typing import List, Dict, Callable, Tuple
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from .tailscale_helpers import get_device_id, update_device_attributes, get_device_attributes, get_tailscale_devices, Device


class TailscaleDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    discovery_interval: int = 5,
    discovery_timeout: int = 30,
    update_interval: int = 15,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    tailscale_api_key: str = None,
    tailnet: str = None,
    allowed_node_ids: List[str] = None,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.create_peer_handle = create_peer_handle
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.update_interval = update_interval
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.discovery_task = None
    self.cleanup_task = None
    self.tailscale_api_key = tailscale_api_key
    self.tailnet = tailnet
    self.allowed_node_ids = allowed_node_ids
    self._device_id = None
    self.update_task = None

  async def start(self):
    self.device_capabilities = await device_capabilities()
    self.discovery_task = asyncio.create_task(self.task_discover_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())
    self.update_task = asyncio.create_task(self.task_update_device_posture_attributes())

  async def task_update_device_posture_attributes(self):
    while True:
      try:
        await self.update_device_posture_attributes()
        if DEBUG_DISCOVERY >= 2:
          print(f"Updated device posture attributes")
      except Exception as e:
        print(f"Error updating device posture attributes: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.update_interval)

  async def get_device_id(self):
    if self._device_id:
      return self._device_id
    self._device_id = await get_device_id()
    return self._device_id

  async def update_device_posture_attributes(self):
    await update_device_attributes(await self.get_device_id(), self.tailscale_api_key, self.node_id, self.node_port, self.device_capabilities)

  async def task_discover_peers(self):
    while True:
      try:
        devices: dict[str, Device] = await get_tailscale_devices(self.tailscale_api_key, self.tailnet)
        current_time = time.time()

        active_devices = {name: device for name, device in devices.items() if device.last_seen is not None and (current_time - device.last_seen.timestamp()) < 30}

        if DEBUG_DISCOVERY >= 4: print(f"Found tailscale devices: {devices}")
        if DEBUG_DISCOVERY >= 2: print(f"Active tailscale devices: {len(active_devices)}/{len(devices)}")
        if DEBUG_DISCOVERY >= 2: print("Time since last seen tailscale devices", [(current_time - device.last_seen.timestamp()) for device in devices.values()])

        for device in active_devices.values():
          if device.name == self.node_id: continue
          peer_host = device.addresses[0]
          peer_id, peer_port, device_capabilities = await get_device_attributes(device.device_id, self.tailscale_api_key)
          if not peer_id:
            if DEBUG_DISCOVERY >= 4: print(f"{device.device_id} does not have exo node attributes. skipping.")
            continue

          if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
            if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
            continue

          if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
            new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", "TS", device_capabilities)
            if not await new_peer_handle.health_check():
              if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
              continue

            if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
            self.known_peers[peer_id] = (
              new_peer_handle,
              current_time,
              current_time,
            )
          else:
            if not await self.known_peers[peer_id][0].health_check():
              if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
              if peer_id in self.known_peers: del self.known_peers[peer_id]
              continue
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
    if self.update_task:
      self.update_task.cancel()
    if self.discovery_task or self.cleanup_task or self.update_task:
      await asyncio.gather(self.discovery_task, self.cleanup_task, self.update_task, return_exceptions=True)

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
        peers_to_remove = []

        peer_ids = list(self.known_peers.keys())
        results = await asyncio.gather(*[self.check_peer(peer_id, current_time) for peer_id in peer_ids], return_exceptions=True)

        for peer_id, should_remove in zip(peer_ids, results):
          if should_remove: peers_to_remove.append(peer_id)

        if DEBUG_DISCOVERY >= 2:
          print(
            "Peer statuses:", {
              peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={await peer_handle.health_check()}, connected_at={connected_at}, last_seen={last_seen}"
              for peer_handle, connected_at, last_seen in self.known_peers.values()
            }
          )

        for peer_id in peers_to_remove:
          if peer_id in self.known_peers:
            del self.known_peers[peer_id]
            if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity or failed health check.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def check_peer(self, peer_id: str, current_time: float) -> bool:
    peer_handle, connected_at, last_seen = self.known_peers.get(peer_id, (None, None, None))
    if peer_handle is None: return False

    try:
      is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error checking peer {peer_id}: {e}")
      return True

    should_remove = ((not is_connected and current_time - connected_at > self.discovery_timeout) or (current_time - last_seen > self.discovery_timeout) or (not health_ok))
    return should_remove
