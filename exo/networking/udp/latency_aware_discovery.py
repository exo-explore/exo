import time
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.inference.shard import Shard

class LatencyAwareDiscovery:
    def __init__(self, node_id, node_port, listen_port, broadcast_port, peer_handle_factory, discovery_timeout):
        self.node_id = node_id
        self.node_port = node_port
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port
        self.peer_handle_factory = peer_handle_factory
        self.discovery_timeout = discovery_timeout
        self.discovery = UDPDiscovery(node_id, node_port, listen_port, broadcast_port, self.create_peer_handle, discovery_timeout)

    async def create_peer_handle(self, peer_id, address, device_capabilities):
        peer_handle = self.peer_handle_factory(peer_id, address, device_capabilities)
        await peer_handle.connect()
        start_time = time.time()
        await peer_handle.health_check()
        end_time = time.time()
        latency = end_time - start_time
        throughput = await self.measure_throughput(peer_handle)
        return peer_handle, latency, throughput

    async def measure_throughput(self, peer_handle):
        # Implement a simple throughput measurement by sending a small payload and measuring the time taken
        payload = b"test" * 1024  # 4KB payload
        start_time = time.time()
        await peer_handle.send_prompt(Shard("test", 0, 1, 1), "test", None, None, None)
        end_time = time.time()
        throughput = len(payload) / (end_time - start_time)
        return throughput

    async def start(self):
        await self.discovery.start()

    async def stop(self):
        await self.discovery.stop()

    async def discover_peers(self, wait_for_peers: int = 0):
        return await self.discovery.discover_peers(wait_for_peers)

    def get_peers(self):
        return self.discovery.get_peers()