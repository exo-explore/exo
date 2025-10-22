import asyncio
import json
from typing import Callable, Dict, List, Optional, Tuple

from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from .radio_transport import RadioTransport, LoopbackHub, LoopbackRadioTransport
from .radio_peer_handle import RadioPeerHandle


class RadioDiscovery(Discovery):
  """Simple discovery over radio: periodically emits beacons with our node_id.
  This reference implementation uses LoopbackHub to simulate a radio channel.
  For real radios, implement a RadioTransport that supports broadcast/unicast.
  """
  def __init__(
    self,
    node_id: str,
    listen_addr: str,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    transport: RadioTransport,
    discovery_interval: float = 2.0,
    discovery_timeout: float = 30.0,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
  ):
    self.node_id = node_id
    self.listen_addr = listen_addr
    self.create_peer_handle = create_peer_handle
    self.transport = transport
    self.discovery_interval = discovery_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self._tasks: List[asyncio.Task] = []

  async def start(self) -> None:
    await self.transport.open()
    self.device_capabilities = await device_capabilities()
    self._tasks = [
      asyncio.create_task(self._beacon_loop()),
      asyncio.create_task(self._listen_loop()),
      asyncio.create_task(self._cleanup_loop()),
    ]

  async def stop(self) -> None:
    for t in self._tasks:
      t.cancel()
    if self._tasks:
      await asyncio.gather(*self._tasks, return_exceptions=True)
    await self.transport.close()

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2:
          print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return [peer for peer, _, _ in self.known_peers.values()]

  async def _beacon_loop(self):
    while True:
      try:
        msg = json.dumps({
          "type": "discovery",
          "node_id": self.node_id,
          "addr": self.listen_addr,
          "device_capabilities": self.device_capabilities.to_dict(),
        }).encode()
        # For this minimal example, emit directly to our known peers (no broadcast)
        # Real radios would use a broadcast address or shared channel
        await self.transport.send(self.listen_addr, msg)
      except Exception:
        pass
      finally:
        await asyncio.sleep(self.discovery_interval)

  async def _listen_loop(self):
    while True:
      src_payload = await self.transport.recv()
      if not src_payload: continue
      src, payload = src_payload
      try:
        data = json.loads(payload.decode())
      except Exception:
        continue
      if data.get("type") != "discovery":
        continue
      peer_id = data.get("node_id")
      if peer_id == self.node_id:
        continue
      peer_addr = data.get("addr")
      cap = DeviceCapabilities(**data.get("device_capabilities", {}))
      if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != peer_addr:
        ph = self.create_peer_handle(peer_id, peer_addr, "RADIO", cap)
        # Note: our RadioPeerHandle needs a RadioTransport; create a per-peer instance if needed
        self.known_peers[peer_id] = (ph, asyncio.get_event_loop().time(), asyncio.get_event_loop().time())
      else:
        self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], asyncio.get_event_loop().time())

  async def _cleanup_loop(self):
    while True:
      now = asyncio.get_event_loop().time()
      to_remove = []
      for peer_id, (_, connected_at, last_seen) in self.known_peers.items():
        if now - last_seen > self.discovery_timeout:
          to_remove.append(peer_id)
      for peer_id in to_remove:
        del self.known_peers[peer_id]
      await asyncio.sleep(self.discovery_interval)
