import unittest
from unittest.mock import Mock, AsyncMock
import numpy as np
import pytest

from .node import Node
from exo.networking.peer_handle import PeerHandle
from exo.download.shard_download import NoopShardDownloader

class TestNode(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.mock_inference_engine = AsyncMock()
    self.mock_server = AsyncMock()
    self.mock_server.start = AsyncMock()
    self.mock_server.stop = AsyncMock()
    self.mock_discovery = AsyncMock()
    self.mock_discovery.start = AsyncMock()
    self.mock_discovery.stop = AsyncMock()
    mock_peer1 = Mock(spec=PeerHandle)
    mock_peer1.id.return_value = "peer1"
    mock_peer2 = Mock(spec=PeerHandle)
    mock_peer2.id.return_value = "peer2"
    self.mock_discovery.discover_peers = AsyncMock(return_value=[mock_peer1, mock_peer2])

    self.node = Node("test_node", self.mock_server, self.mock_inference_engine, "localhost", 50051, self.mock_discovery, NoopShardDownloader())

  async def asyncSetUp(self):
    await self.node.start()

  async def asyncTearDown(self):
    await self.node.stop()

  async def test_node_initialization(self):
    self.assertEqual(self.node.node_id, "test_node")
    self.assertEqual(self.node.host, "localhost")
    self.assertEqual(self.node.port, 50051)

  async def test_node_start(self):
    self.mock_server.start.assert_called_once_with("localhost", 50051)

  async def test_node_stop(self):
    await self.node.stop()
    self.mock_server.stop.assert_called_once()

  async def test_discover_and_connect_to_peers(self):
    await self.node.discover_and_connect_to_peers()
    self.assertEqual(len(self.node.peers), 2)
    self.assertIn("peer1", map(lambda p: p.id(), self.node.peers))
    self.assertIn("peer2", map(lambda p: p.id(), self.node.peers))

  async def test_process_tensor_calls_inference_engine(self):
    mock_peer = Mock()
    self.node.peers = [mock_peer]

    input_tensor = np.array([69, 1, 2])
    await self.node.process_tensor(input_tensor, None)

    self.node.inference_engine.process_shard.assert_called_once_with(input_tensor)

  @pytest.mark.asyncio
  async def test_node_capabilities():
    node = Node()
    await node.initialize()
    caps = await node.get_device_capabilities()
    assert caps is not None
    assert caps.model != ""
