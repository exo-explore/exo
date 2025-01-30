import asyncio
import json
import socket
import time
import traceback
from typing import List, Dict, Callable, Tuple, Coroutine, Optional
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses_and_interfaces, get_interface_priority_and_type


class ListenProtocol(asyncio.DatagramProtocol):
  def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
    super().__init__()
    self.on_message = on_message
    self.loop = asyncio.get_event_loop()

  def connection_made(self, transport):
    self.transport = transport

  def datagram_received(self, data, addr):
    asyncio.create_task(self.on_message(data, addr))


def get_broadcast_address(ip_addr: str) -> str:
  try:
    # Split IP into octets and create broadcast address for the subnet
    ip_parts = ip_addr.split('.')
    return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.255"
  except:
    return "255.255.255.255"


class BroadcastProtocol(asyncio.DatagramProtocol):
  def __init__(self, message: str, broadcast_port: int, source_ip: str):
    self.message = message
    self.broadcast_port = broadcast_port
    self.source_ip = source_ip

  def connection_made(self, transport):
    sock = transport.get_extra_info("socket")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # Try both subnet-specific and global broadcast
    broadcast_addr = get_broadcast_address(self.source_ip)
    transport.sendto(self.message.encode("utf-8"), (broadcast_addr, self.broadcast_port))
    if broadcast_addr != "255.255.255.255":
      transport.sendto(self.message.encode("utf-8"), ("255.255.255.255", self.broadcast_port))


class UDPDiscovery(Discovery):
  def __init__(
    self,
    node_id: str,
    node_port: int,
    listen_port: int,
    broadcast_port: int,
    create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
    broadcast_interval: int = 2.5,
    discovery_timeout: int = 30,
    device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
    allowed_node_ids: Optional[List[str]] = None,
    allowed_interface_types: Optional[List[str]] = None,
  ):
    self.node_id = node_id
    self.node_port = node_port
    self.listen_port = listen_port
    self.broadcast_port = broadcast_port
    self.create_peer_handle = create_peer_handle
    self.broadcast_interval = broadcast_interval
    self.discovery_timeout = discovery_timeout
    self.device_capabilities = device_capabilities
    self.allowed_node_ids = allowed_node_ids
    self.allowed_interface_types = allowed_interface_types
    self.known_peers: Dict[str, Tuple[PeerHandle, float, float, int]] = {}
    self.broadcast_task = None
    self.listen_task = None
    self.cleanup_task = None

  async def start(self):
    self.device_capabilities = await device_capabilities()
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
    return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]

  async def task_broadcast_presence(self):
    while True:
      for addr, interface_name in get_all_ip_addresses_and_interfaces():
        interface_priority, interface_type = await get_interface_priority_and_type(interface_name)
        message = json.dumps({
          "type": "discovery",
          "node_id": self.node_id,
          "grpc_port": self.node_port,
          "device_capabilities": self.device_capabilities.to_dict(),
          "priority": interface_priority,
          "interface_name": interface_name,
          "interface_type": interface_type,
        })

        transport = None
        try:
          sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
          sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
          sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
          try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
          except AttributeError:
            pass
          sock.bind((addr, 0))
          
          transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
            lambda: BroadcastProtocol(message, self.broadcast_port, addr),
            sock=sock
          )
        except Exception as e:
          print(f"Error in broadcast presence ({addr} - {interface_name} - {interface_priority}): {e}")
        finally:
          if transport:
            try: transport.close()
            except Exception as e:
              if DEBUG_DISCOVERY >= 2: print(f"Error closing transport: {e}")

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
      
      # Skip if peer_id is not in allowed list
      if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
        if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
        return

      peer_host = addr[0]
      peer_port = message["grpc_port"]
      peer_prio = message["priority"]
      peer_interface_name = message["interface_name"]
      peer_interface_type = message["interface_type"]

      # Skip if interface type is not in allowed list
      if self.allowed_interface_types and peer_interface_type not in self.allowed_interface_types:
        if DEBUG_DISCOVERY >= 2: print(f"Ignoring peer {peer_id} as its interface type {peer_interface_type} is not in the allowed interface types list")
        return

      device_capabilities = DeviceCapabilities(**message["device_capabilities"])

      if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
        if peer_id in self.known_peers:
          existing_peer_prio = self.known_peers[peer_id][3]
          if existing_peer_prio >= peer_prio:
            if DEBUG >= 1:
              print(f"Ignoring peer {peer_id} at {peer_host}:{peer_port} with priority {peer_prio} because we already know about a peer with higher or equal priority: {existing_peer_prio}")
            return
        new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", f"{peer_interface_type} ({peer_interface_name})", device_capabilities)
        if not await new_peer_handle.health_check():
          if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
          return
        if DEBUG >= 1: print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
        self.known_peers[peer_id] = (new_peer_handle, time.time(), time.time(), peer_prio)
      else:
        if not await self.known_peers[peer_id][0].health_check():
          if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
          if peer_id in self.known_peers: del self.known_peers[peer_id]
          return
        if peer_id in self.known_peers: self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], time.time(), peer_prio)

  async def task_listen_for_peers(self):
    await asyncio.get_event_loop().create_datagram_endpoint(lambda: ListenProtocol(self.on_listen_message), local_addr=("0.0.0.0", self.listen_port))
    if DEBUG_DISCOVERY >= 2: print("Started listen task")

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
              peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={await peer_handle.health_check()}, connected_at={connected_at}, last_seen={last_seen}, prio={prio}"
              for peer_handle, connected_at, last_seen, prio in self.known_peers.values()
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
        await asyncio.sleep(self.broadcast_interval)

  async def check_peer(self, peer_id: str, current_time: float) -> bool:
    peer_handle, connected_at, last_seen, prio = self.known_peers.get(peer_id, (None, None, None, None))
    if peer_handle is None: return False

    try:
      is_connected = await peer_handle.is_connected()
      health_ok = await peer_handle.health_check()
    except Exception as e:
      if DEBUG_DISCOVERY >= 2: print(f"Error checking peer {peer_id}: {e}")
      return True

    should_remove = ((not is_connected and current_time - connected_at > self.discovery_timeout) or (current_time - last_seen > self.discovery_timeout) or (not health_ok))
    return should_remove
