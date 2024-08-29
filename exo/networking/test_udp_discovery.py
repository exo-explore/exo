import asyncio
import unittest
from unittest import mock  # Add this import
from exo.networking.udp_discovery import UDPDiscovery

class TestUDPDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.peer1 = mock.AsyncMock()
    self.peer2 = mock.AsyncMock()
    self.peer1.connect = mock.AsyncMock()
    self.peer2.connect = mock.AsyncMock()
    self.discovery1 = UDPDiscovery("discovery1", 50051, 5678, 5679, create_peer_handle=lambda peer_id, address, device_capabilities: self.peer1)
    self.discovery2 = UDPDiscovery("discovery2", 50052, 5679, 5678, create_peer_handle=lambda peer_id, address, device_capabilities: self.peer2)
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

if __name__ == "__main__":
  asyncio.run(unittest.main())
