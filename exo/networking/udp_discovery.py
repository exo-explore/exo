import asyncio
import json
import socket
import time
import traceback
from typing import List, Dict, Callable, Tuple, Coroutine
from .discovery import Discovery
from .peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses

class ListenProtocol(asyncio.DatagramProtocol):
  def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
    super().__init__()
    self.on_message = on_message
    self.loop = asyncio.get_event_loop()

  def connection_made(self, transport):
    self.transport = transport

  def datagram_received(self, data, addr):
    asyncio.create_task(self.on_message(data, addr))


class BroadcastProtocol(asyncio.DatagramProtocol):
  def __init__(self, message: str, broadcast_port: int):
    self.message = message
    self.broadcast_port = broadcast_port

  def connection_made(self, transport):
    sock = transport.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    transport.sendto(self.message.encode("utf-8"), ("<broadcast>", self.broadcast_port))


class UDPDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    listen_port: int,
    broadcast_port: int,
    create_peer_handle: Callable[[str, str, DeviceCapabilities], PeerHandle],
    broadcast_interval: int = 1,
    discovery_timeout: int = 30,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.listen_port = listen_port
    self.broadcast_port = broadcast_port
    self.create_peer_handle = create_peer_handle
    self.broadcast_interval = broadcast_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
    self.broadcast_task = None
    self.listen_task = None
    self.cleanup_task = None

  async def start(self):
    self.device_capabilities = device_capabilities()
    self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())
    self.listen_task = asyncio.create_task(self.task_listen_for_peers())
    self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

  async def stop(self):
    if self.broadcast_task: self.broadcast_task.cancel()
    if self.listen_task: self.listen_task.cancel()
    if self.cleanup_task: self.cleanup_task.cancel()
    if self.broadcast_task or self.listen_task or self.cleanup_task:
      await asyncio.gather(self.broadcast_task, self.listen_task, self.cleanup_task, return_exceptions=True)

  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    if wait_for_peers > 0:
      while len(self.known_peers) < wait_for_peers:
        if DEBUG_DISCOVERY >= 2: print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
        await asyncio.sleep(0.1)
    return [peer_handle for peer_handle, _, _ in self.known_peers.values()]

  async def task_broadcast_presence(self):
    message = json.dumps({
      "type": "discovery",
      "node_id": self.node_id,
      "grpc_port": self.node_port,
      "device_capabilities": self.device_capabilities.to_dict(),
    })

    if DEBUG_DISCOVERY >= 2:
      print("Starting task_broadcast_presence...")
      print(f"\nBroadcast message: {message}")

    while True:
      # Explicitly broadcasting on all assigned ips since broadcasting on `0.0.0.0` on MacOS does not broadcast over
      # the Thunderbolt bridge when other connection modalities exist such as WiFi or Ethernet
      for addr in get_all_ip_addresses():
        transport = None
        try:
          transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
            lambda: BroadcastProtocol(message, self.broadcast_port),
            local_addr=(addr, 0),
            family=socket.AF_INET
          )
          if DEBUG_DISCOVERY >= 3:
            print(f"Broadcasting presence at ({addr})")
        except Exception as e:
          print(f"Error in broadcast presence ({addr}): {e}")
        finally:
          if transport:
            try:
              transport.close()
            except Exception as e:
              if DEBUG_DISCOVERY >= 2: print(f"Error closing transport: {e}")
              if DEBUG_DISCOVERY >= 2: traceback.print_exc()
      await asyncio.sleep(self.broadcast_interval)

  async def on_listen_message(self, data, addr):
    if not data:
      return

    decoded_data = data.decode("utf-8", errors="ignore")

    # Check if the decoded data starts with a valid JSON character
    if not (decoded_data.strip() and decoded_data.strip()[0] in "{["):
      if DEBUG_DISCOVERY >= 2: print(f"Received invalid JSON data from {addr}: {decoded_data[:100]}")
      return

    try:
      decoder = json.JSONDecoder(strict=False)
      message = decoder.decode(decoded_data)
    except json.JSONDecodeError as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error decoding JSON data from {addr}: {e}")
      return

    if DEBUG_DISCOVERY >= 2: print(f"received from peer {addr}: {message}")

    if message["type"] == "discovery" and message["node_id"] != self.node_id:
      peer_id = message["node_id"]
      peer_host = addr[0]
      peer_port = message["grpc_port"]
      device_capabilities = DeviceCapabilities(**message["device_capabilities"])
      if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
        if DEBUG >= 1: print(
          f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
        self.known_peers[peer_id] = (
          self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", device_capabilities),
          time.time(),
          time.time(),
        )
      self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], time.time())

  async def task_listen_for_peers(self):
    await asyncio.get_event_loop().create_datagram_endpoint(lambda: ListenProtocol(self.on_listen_message),
                                                            local_addr=("0.0.0.0", self.listen_port))
    if DEBUG_DISCOVERY >= 2: print("Started listen task")

  async def task_cleanup_peers(self):
    while True:
      try:
        current_time = time.time()
        peers_to_remove = [
          peer_handle.id() for peer_handle, connected_at, last_seen in self.known_peers.values()
          if (not await peer_handle.is_connected() and current_time - connected_at > self.discovery_timeout) or current_time - last_seen > self.discovery_timeout
        ]
        if DEBUG_DISCOVERY >= 2: print("Peer statuses:", {peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, {connected_at=}, {last_seen=}" for peer_handle, connected_at, last_seen in self.known_peers.values()})
        for peer_id in peers_to_remove:
          if peer_id in self.known_peers: del self.known_peers[peer_id]
          if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity.")
      except Exception as e:
        print(f"Error in cleanup peers: {e}")
        print(traceback.format_exc())
      finally:
        await asyncio.sleep(self.broadcast_interval)
