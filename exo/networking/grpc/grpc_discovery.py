import asyncio
import json
import socket
import time
import netifaces
import psutil
import subprocess
from typing import List, Dict, Callable, Tuple, Coroutine
from ..discovery import Discovery
from ..peer_handle import PeerHandle
from .grpc_peer_handle import GRPCPeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo import DEBUG_DISCOVERY

def get_interface_priority(interface_name, interface_type='', hardware=''):
    if 'thunderbolt' in hardware or 'thunderbolt' in interface_type:
        return 3
    elif interface_type == 'wi-fi' or 'airport' in hardware:
        return 2
    elif interface_name.startswith('en'):
        return 1
    else:
        return 0

def get_network_interfaces():
    if psutil.MACOS:
        network_info = subprocess.check_output(['system_profiler', 'SPNetworkDataType', '-json']).decode('utf-8')
        network_data = json.loads(network_info)

        interfaces = []
        for interface in network_data.get('SPNetworkDataType', []):
            interface_name = interface.get('interface')
            interface_type = interface.get('type', '').lower()
            hardware = interface.get('hardware', '').lower()

            if not interface_name:
                continue

            priority = get_interface_priority(interface_name, interface_type, hardware)
            interfaces.append((interface_name, priority))

        return sorted(interfaces, key=lambda x: x[1], reverse=True)
    else:
        return [(iface, get_interface_priority(iface)) for iface in netifaces.interfaces()]

class ListenProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
        super().__init__()
        self.on_message = on_message
        self.loop = asyncio.get_event_loop()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        asyncio.create_task(self.on_message(data, addr))

class GRPCDiscovery(Discovery):
    def __init__(self, node_id: str, node_port: int, listen_port: int, broadcast_port: int = None, broadcast_interval: int = 1, device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES):
        self.node_id = node_id
        self.node_port = node_port
        self.device_capabilities = device_capabilities
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port if broadcast_port is not None else listen_port
        self.broadcast_interval = broadcast_interval
        self.known_peers: Dict[str, Tuple[GRPCPeerHandle, float, float, int]] = {}
        self.broadcast_task = None
        self.listen_task = None
        self.cleanup_task = None

    async def start(self):
        self.device_capabilities = device_capabilities()
        self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())
        self.listen_task = asyncio.create_task(self.task_listen_for_peers())
        self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

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
        if DEBUG_DISCOVERY >= 2: print("Starting peer discovery process...")

        if wait_for_peers > 0:
            while len(self.known_peers) == 0:
                if DEBUG_DISCOVERY >= 2: print("No peers discovered yet, retrying in 1 second...")
                await asyncio.sleep(1)  # Keep trying to find peers
            if DEBUG_DISCOVERY >= 2: print(f"Discovered first peer: {next(iter(self.known_peers.values()))}")

        grace_period = 5  # seconds
        while True:
            initial_peer_count = len(self.known_peers)
            if DEBUG_DISCOVERY >= 2: print(f"Current number of known peers: {initial_peer_count}. Waiting {grace_period} seconds to discover more...")
            if len(self.known_peers) == initial_peer_count:
                if wait_for_peers > 0:
                    await asyncio.sleep(grace_period)
                    if DEBUG_DISCOVERY >= 2: print(f"Waiting additional {wait_for_peers} seconds for more peers.")
                    wait_for_peers = 0
                else:
                    if DEBUG_DISCOVERY >= 2: print("No new peers discovered in the last grace period. Ending discovery process.")
                    break  # No new peers found in the grace period, we are done

        return [peer_handle for peer_handle, _, _, _ in self.known_peers.values()]

    async def task_broadcast_presence(self):
        while True:
            try:
                interfaces = get_network_interfaces()
                
                for interface, priority in interfaces:
                    try:
                        if DEBUG_DISCOVERY >= 3: print(f"Broadcasting on interface: {interface} with priority: {priority}")
                        message = json.dumps({
                            "type": "discovery",
                            "node_id": self.node_id,
                            "grpc_port": self.node_port,
                            "device_capabilities": self.device_capabilities.to_dict(),
                            "priority": priority
                        }).encode('utf-8')
                        
                        transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                            lambda: asyncio.DatagramProtocol(),
                            local_addr=(netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr'], 0),
                            family=socket.AF_INET)
                        sock = transport.get_extra_info('socket')
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                        transport.sendto(message, ('<broadcast>', self.broadcast_port))
                        transport.close()
                    except Exception as e:
                        if DEBUG_DISCOVERY >= 2: print(f"Error broadcasting on interface {interface}: {e}")
            except Exception as e:
                if DEBUG_DISCOVERY >= 2: print(f"Error updating or accessing interfaces: {e}")
            
            await asyncio.sleep(self.broadcast_interval)

    async def on_listen_message(self, data, addr):
        message = json.loads(data.decode('utf-8'))
        if DEBUG_DISCOVERY >= 2: print(f"received from peer {addr}: {message}")
        if message['type'] == 'discovery' and message['node_id'] != self.node_id:
            peer_id = message['node_id']
            peer_host = addr[0]
            peer_port = message['grpc_port']
            new_priority = message['priority']
            device_capabilities = DeviceCapabilities(**message['device_capabilities'])
            
            if peer_id in self.known_peers:
                _, _, _, current_priority = self.known_peers[peer_id]
                if new_priority > current_priority:
                    await self.reconnect_peer(peer_id, peer_host, peer_port, new_priority, device_capabilities)
            else:
                peer_handle = GRPCPeerHandle(peer_id, f"{peer_host}:{peer_port}", device_capabilities)
                self.known_peers[peer_id] = (peer_handle, time.time(), time.time(), new_priority)
                if DEBUG_DISCOVERY >= 2: print(f"Discovered new peer {peer_id} at {peer_host}:{peer_port} with priority {new_priority}")

    async def reconnect_peer(self, peer_id, new_host, new_port, new_priority, device_capabilities):
        old_handle, connected_at, _, _ = self.known_peers[peer_id]
        await old_handle.disconnect()
        new_handle = GRPCPeerHandle(peer_id, f"{new_host}:{new_port}", device_capabilities)
        await new_handle.connect()
        self.known_peers[peer_id] = (new_handle, connected_at, time.time(), new_priority)
        if DEBUG_DISCOVERY >= 2: print(f"Reconnected to peer {peer_id} at {new_host}:{new_port} with new priority {new_priority}")

    async def task_listen_for_peers(self):
        try:
            if DEBUG_DISCOVERY >= 2: print("Starting to listen on all interfaces")
            await asyncio.get_event_loop().create_datagram_endpoint(
                lambda: ListenProtocol(self.on_listen_message),
                local_addr=('0.0.0.0', self.listen_port)
            )
            if DEBUG_DISCOVERY >= 2: print("Started listen task on all interfaces")
        except Exception as e:
            if DEBUG_DISCOVERY >= 2: print(f"Error setting up listening: {e}")

    async def task_cleanup_peers(self):
        while True:
            try:
                current_time = time.time()
                timeout = 15 * self.broadcast_interval
                peers_to_remove = [
                    peer_handle.id() for peer_handle, connected_at, last_seen, _ in self.known_peers.values() if
                    (not await peer_handle.is_connected() and current_time - connected_at > timeout) or current_time - last_seen > timeout
                ]
                if DEBUG_DISCOVERY >= 2: print("Peer statuses:", {peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, {connected_at=}, {last_seen=}, priority={priority}" for peer_handle, connected_at, last_seen, priority in self.known_peers.values()})
                if DEBUG_DISCOVERY >= 2 and len(peers_to_remove) > 0: print(f"Cleaning up peers: {peers_to_remove}")
                for peer_id in peers_to_remove:
                    if peer_id in self.known_peers: del self.known_peers[peer_id]
                    if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity.")
                await asyncio.sleep(self.broadcast_interval)
            except Exception as e:
                print(f"Error in cleanup peers: {e}")
                import traceback
                print(traceback.format_exc())
