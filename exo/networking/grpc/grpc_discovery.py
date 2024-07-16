import asyncio
import json
import socket
import time
from typing import List, Dict
from ..discovery import Discovery
from ..peer_handle import PeerHandle
from .grpc_peer_handle import GRPCPeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities
from exo import DEBUG

class GRPCDiscovery(Discovery):
    def __init__(self, node_id: str, node_port: int, listen_port: int, broadcast_port: int = None, broadcast_interval: int = 1, device_capabilities=None):
        self.node_id = node_id
        self.node_port = node_port
        self.device_capabilities = device_capabilities
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port if broadcast_port is not None else listen_port
        self.broadcast_interval = broadcast_interval
        self.known_peers: Dict[str, GRPCPeerHandle] = {}
        self.peer_last_seen: Dict[str, float] = {}
        self.broadcast_task = None
        self.listen_task = None
        self.cleanup_task = None

    async def start(self):
        self.broadcast_task = asyncio.create_task(self._broadcast_presence())
        self.listen_task = asyncio.create_task(self._listen_for_peers())
        self.cleanup_task = asyncio.create_task(self._cleanup_peers())

    async def stop(self):
        if self.broadcast_task:
            self.broadcast_task.cancel()
        if self.listen_task:
            self.listen_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.broadcast_task or self.listen_task or self.cleanup_task:
            await asyncio.gather(self.broadcast_task, self.listen_task, self.cleanup_task, return_exceptions=True)

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        if DEBUG >= 2: print("Starting peer discovery process...")

        if wait_for_peers > 0:
            while len(self.known_peers) == 0:
                if DEBUG >= 2: print("No peers discovered yet, retrying in 1 second...")
                await asyncio.sleep(1)  # Keep trying to find peers
            if DEBUG >= 2: print(f"Discovered first peer: {next(iter(self.known_peers.values()))}")

        grace_period = 5  # seconds
        while True:
            initial_peer_count = len(self.known_peers)
            if DEBUG >= 2: print(f"Current number of known peers: {initial_peer_count}. Waiting {grace_period} seconds to discover more...")
            await asyncio.sleep(grace_period)
            if len(self.known_peers) == initial_peer_count:
                if wait_for_peers > 0:
                    if DEBUG >= 2: print(f"Waiting additional {wait_for_peers} seconds for more peers.")
                    await asyncio.sleep(wait_for_peers)
                    wait_for_peers = 0
                else:
                    if DEBUG >= 2: print("No new peers discovered in the last grace period. Ending discovery process.")
                    break  # No new peers found in the grace period, we are done

        return list(self.known_peers.values())

    async def _broadcast_presence(self):
        if not self.device_capabilities:
            self.device_capabilities = device_capabilities()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        message = json.dumps({
            "type": "discovery",
            "node_id": self.node_id,
            "grpc_port": self.node_port,
            "device_capabilities": {
                "model": self.device_capabilities.model,
                "chip": self.device_capabilities.chip,
                "memory": self.device_capabilities.memory
            }
        }).encode('utf-8')

        while True:
            sock.sendto(message, ('<broadcast>', self.broadcast_port))
            await asyncio.sleep(self.broadcast_interval)

    async def _listen_for_peers(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.listen_port))
        sock.setblocking(False)

        while True:
            try:
                data, addr = await asyncio.get_event_loop().sock_recvfrom(sock, 1024)
                message = json.loads(data.decode('utf-8'))
                if DEBUG >= 2: print(f"received from peer {addr}: {message}")
                if message['type'] == 'discovery' and message['node_id'] != self.node_id:
                    peer_id = message['node_id']
                    peer_host = addr[0]
                    peer_port = message['grpc_port']
                    device_capabilities = DeviceCapabilities(**message['device_capabilities'])
                    if peer_id not in self.known_peers:
                        self.known_peers[peer_id] = GRPCPeerHandle(peer_id, f"{peer_host}:{peer_port}", device_capabilities)
                        if DEBUG >= 2: print(f"Discovered new peer {peer_id} at {peer_host}:{peer_port}")
                    self.peer_last_seen[peer_id] = time.time()
            except Exception as e:
                print(f"Error in peer discovery: {e}")
                import traceback
                print(traceback.format_exc())
                await asyncio.sleep(self.broadcast_interval / 2)

    async def _cleanup_peers(self):
        while True:
            current_time = time.time()
            timeout = 15 * self.broadcast_interval
            peers_to_remove = [peer_id for peer_id, last_seen in self.peer_last_seen.items() if current_time - last_seen > timeout]
            for peer_id in peers_to_remove:
                del self.known_peers[peer_id]
                del self.peer_last_seen[peer_id]
                if DEBUG >= 2: print(f"Removed peer {peer_id} due to inactivity.")
            await asyncio.sleep(self.broadcast_interval)
