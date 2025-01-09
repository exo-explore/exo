import json
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
    self.discovery1 = ManualDiscovery(
      root_path,
      "node1",
      create_peer_handle=lambda peer_id, address, description, device_capabilities: self.peer1,
    )
    await self.discovery1.start()

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
    self.discovery1 = ManualDiscovery(
      root_path,
      "node1",
      create_peer_handle=lambda peer_id, address, description, device_capabilities: self.peer1,
    )
    self.discovery2 = ManualDiscovery(
      root_path,
      "node2",
      create_peer_handle=lambda peer_id, address, description, device_capabilities: self.peer2,
    )
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
    self.discovery1 = ManualDiscovery(
      root_path,
      "node1",
      create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    )
    self.discovery2 = ManualDiscovery(
      root_path,
      "node2",
      create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    )
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

  async def test_dynamic_config_update(self):
    initial_peers = await self.discovery1.discover_peers(wait_for_peers=1)
    self.assertEqual(len(initial_peers), 1)

    # Save original config for cleanup
    with open(root_path, "r") as f:
      original_config = json.load(f)

    try:
      updated_config = {
        "peers": {
          **original_config["peers"],
          "node3": {
            "address": "localhost",
            "port": 50053,
            "device_capabilities": {
              "model": "Unknown Model",
              "chip": "Unknown Chip",
              "memory": 0,
              "flops": {"fp32": 0, "fp16": 0, "int8": 0},
            },
          },
        }
      }

      with open(root_path, "w") as f:
        json.dump(updated_config, f, indent=2)

      node3 = mock.AsyncMock(spec=Node)
      server3 = GRPCServer(node3, "localhost", 50053)
      await server3.start()

      try:
        # Wait for the config to be reloaded
        await asyncio.sleep(1.5)

        updated_peers = await self.discovery1.discover_peers(wait_for_peers=2)
        self.assertEqual(len(updated_peers), 2)

        for peer in updated_peers:
          await peer.connect()
          self.assertTrue(await peer.is_connected())

      finally:
        await server3.stop()

    finally:
      # Restore the original config file
      with open(root_path, "w") as f:
        json.dump(original_config, f, indent=2)

    # Wait for the config to be reloaded again
    await asyncio.sleep(1.5)

    updated_peers = await self.discovery1.discover_peers(wait_for_peers=1)
    self.assertEqual(len(updated_peers), 1)


if __name__ == "__main__":
  asyncio.run(unittest.main())
