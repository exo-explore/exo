import asyncio
import unittest
from unittest import mock
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.networking.manual.network_topology_config import NetworkTopology
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.grpc.grpc_server import GRPCServer
from exo.orchestration.node import Node

root_path = "./exo/networking/manual/test_data/test_config.json"


class TestSingleNodeManualDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.peer1 = mock.AsyncMock()
    self.peer1.connect = mock.AsyncMock()
    self.discovery1 = ManualDiscovery(root_path, "node1", create_peer_handle=lambda peer_id, address, device_capabilities: self.peer1)
    _ = self.discovery1.start()

  async def asyncTearDown(self):
    await self.discovery1.stop()

  async def test_discovery(self):
    peers1 = await self.discovery1.discover_peers(wait_for_peers=0)
    assert len(peers1) == 0

    self.peer1.connect.assert_not_called()


class TestManualDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.peer1 = mock.AsyncMock()
    self.peer2 = mock.AsyncMock()
    self.peer1.connect = mock.AsyncMock()
    self.peer2.connect = mock.AsyncMock()
    self.discovery1 = ManualDiscovery(root_path, "node1", create_peer_handle=lambda peer_id, address, device_capabilities: self.peer1)
    self.discovery2 = ManualDiscovery(root_path, "node2", create_peer_handle=lambda peer_id, address, device_capabilities: self.peer2)
    await self.discovery1.start()
    await self.discovery2.start()

  async def asyncTearDown(self):
    await self.discovery1.stop()
    await self.discovery2.stop()

  async def test_discovery(self):
    peers1 = await self.discovery1.discover_peers(wait_for_peers=1)
    assert len(peers1) == 1
    peers2 = await self.discovery2.discover_peers(wait_for_peers=1)
    assert len(peers2) == 1

    # connect has to be explicitly called after discovery
    self.peer1.connect.assert_not_called()
    self.peer2.connect.assert_not_called()


class TestManualDiscoveryWithGRPCPeerHandle(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    config = NetworkTopology.from_path(root_path)

    self.node1 = mock.AsyncMock(spec=Node)
    self.node2 = mock.AsyncMock(spec=Node)
    self.server1 = GRPCServer(self.node1, config.peers["node1"].address, config.peers["node1"].port)
    self.server2 = GRPCServer(self.node2, config.peers["node2"].address, config.peers["node2"].port)
    await self.server1.start()
    await self.server2.start()
    self.discovery1 = ManualDiscovery(root_path, "node1", create_peer_handle=lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities))
    self.discovery2 = ManualDiscovery(root_path, "node2", create_peer_handle=lambda peer_id, address, device_capabilities: GRPCPeerHandle(peer_id, address, device_capabilities))
    await self.discovery1.start()
    await self.discovery2.start()

  async def asyncTearDown(self):
    await self.discovery1.stop()
    await self.discovery2.stop()
    await self.server1.stop()
    await self.server2.stop()

  async def test_grpc_discovery(self):
    peers1 = await self.discovery1.discover_peers(wait_for_peers=1)
    assert len(peers1) == 1
    peers2 = await self.discovery2.discover_peers(wait_for_peers=1)
    assert len(peers2) == 1

    # Connect
    await peers1[0].connect()
    await peers2[0].connect()
    self.assertTrue(await peers1[0].is_connected())
    self.assertTrue(await peers2[0].is_connected())

    # Kill server1
    await self.server1.stop()

    self.assertTrue(await peers1[0].is_connected())
    self.assertFalse(await peers2[0].is_connected())

    # Kill server2
    await self.server2.stop()

    self.assertFalse(await peers1[0].is_connected())
    self.assertFalse(await peers2[0].is_connected())


if __name__ == "__main__":
  asyncio.run(unittest.main())
