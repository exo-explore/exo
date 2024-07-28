import asyncio
import unittest
from .grpc_discovery import GRPCDiscovery


class TestGRPCDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.node1 = GRPCDiscovery("node1", 50051, 5678, 5679)
    self.node2 = GRPCDiscovery("node2", 50052, 5679, 5678)
    await self.node1.start()
    await self.node2.start()

  async def asyncTearDown(self):
    await self.node1.stop()
    await self.node2.stop()

  async def test_discovery(self):
    await asyncio.sleep(4)

    # Check discovered peers
    print("Node1 Peers:", ", ".join([f"{peer_id}: {peer}" for peer_id, peer in self.node1.known_peers.items()]))
    print("Node2 Peers:", ", ".join([f"{peer_id}: {peer}" for peer_id, peer in self.node2.known_peers.items()]))
