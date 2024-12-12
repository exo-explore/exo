import os
import asyncio
import unittest
from unittest import mock
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.peer_handle import PeerHandle


class TestTailscaleDiscovery(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.tailscale_api_key = os.environ.get("TAILSCALE_API_KEY", "")
    self.tailnet = os.environ.get("TAILSCALE_TAILNET", "")
    self.discovery = TailscaleDiscovery(
      node_id="test_node",
      node_port=50051,
      create_peer_handle=lambda peer_id, address, description, device_capabilities: unittest.mock.Mock(spec=PeerHandle, id=lambda: peer_id),
      tailscale_api_key=self.tailscale_api_key,
      tailnet=self.tailnet
    )
    await self.discovery.start()

  async def asyncTearDown(self):
    await self.discovery.stop()

  async def test_discovery(self):
    # Wait for a short period to allow discovery to happen
    await asyncio.sleep(15)

    # Get discovered peers
    peers = await self.discovery.discover_peers()

    # Check if any peers were discovered
    self.assertGreater(len(peers), 0, "No peers were discovered")

    # Print discovered peers for debugging
    print(f"Discovered peers: {[peer.id() for peer in peers]}")

    # Check if discovered peers are instances of GRPCPeerHandle
    print(peers)


if __name__ == '__main__':
  unittest.main()
