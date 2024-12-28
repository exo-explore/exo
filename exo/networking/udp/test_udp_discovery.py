import asyncio
import unittest
from unittest import mock
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.grpc.grpc_server import GRPCServer
from exo.orchestration.node import Node


class TestUDPDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.peer1 = mock.AsyncMock()
    self.peer2 = mock.AsyncMock()
    self.peer1.connect = mock.AsyncMock()
    self.peer2.connect = mock.AsyncMock()
    self.discovery1 = UDPDiscovery("discovery1", 50051, 5678, 5679, create_peer_handle=lambda peer_id, address, description, device_capabilities: self.peer1)
    self.discovery2 = UDPDiscovery("discovery2", 50052, 5679, 5678, create_peer_handle=lambda peer_id, address, description, device_capabilities: self.peer2)
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


class TestUDPDiscoveryWithGRPCPeerHandle(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.node1 = mock.AsyncMock(spec=Node)
    self.node2 = mock.AsyncMock(spec=Node)
    self.server1 = GRPCServer(self.node1, "localhost", 50053)
    self.server2 = GRPCServer(self.node2, "localhost", 50054)
    await self.server1.start()
    await self.server2.start()
    self.discovery1 = UDPDiscovery("discovery1", 50053, 5678, 5679, lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
    self.discovery2 = UDPDiscovery("discovery2", 50054, 5679, 5678, lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
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
    assert not await peers1[0].is_connected()
    assert not await peers2[0].is_connected()

    # Connect
    await peers1[0].connect()
    await peers2[0].connect()
    assert await peers1[0].is_connected()
    assert await peers2[0].is_connected()

    # Kill server1
    await self.server1.stop()

    assert await peers1[0].is_connected()
    assert not await peers2[0].is_connected()


if __name__ == "__main__":
  asyncio.run(unittest.main())
