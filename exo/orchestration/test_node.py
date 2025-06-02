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

import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio # Required for IsolatedAsyncioTestCase if not already imported

# Assuming Node and other necessary imports are already in original_test_node_content
# or will be added if this were a standalone file. For appending, we rely on existing imports.
from exo.orchestration.node import Node
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.inference.inference_engine import InferenceEngine, get_inference_engine
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.inference.llama_cpp.llama_cpp_inference_engine import LlamaCppInferenceEngine
from exo.inference.vllm.vllm_inference_engine import VLLMInferenceEngine
from exo.inference.openvino.openvino_inference_engine import OpenVINOInferenceEngine
from exo.download.shard_download import ShardDownloader, NoopShardDownloader


# Helper to create a DeviceCapabilities object
def create_mock_caps(chip="Unknown CPU", model="TestModel", memory=16000, available_memory=16000, flops_fp32=1.0):
    return DeviceCapabilities(
        chip=chip,
        model=model,
        memory=memory,
        available_memory=available_memory,
        flops=DeviceFlops(fp32=flops_fp32, fp16=flops_fp32*2, int8=flops_fp32*4)
    )

class TestSelectBestInferenceEngine(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_server = AsyncMock()
        self.mock_discovery = AsyncMock()
        # self.mock_shard_downloader = MagicMock(spec=ShardDownloader)
        self.mock_shard_downloader = NoopShardDownloader() # Using NoopShardDownloader

        # Start with a generic mock engine or a Dummy one
        self.initial_engine = MagicMock(spec=InferenceEngine)
        self.initial_engine.__class__.__name__ = "InitialMockEngine"

        self.node = Node(
            _id="test_select_node",
            server=self.mock_server,
            inference_engine=self.initial_engine, # Start with a basic mock
            discovery=self.mock_discovery,
            shard_downloader=self.mock_shard_downloader
        )
        # self.node.device_capabilities is set in Node.start(), so we'll mock it per test
        # or set it directly after Node instantiation for tests that don't call node.start()

    async def asyncSetUp(self):
        # We are testing select_best_inference_engine directly, so full node start might not be needed,
        # but device_capabilities needs to be set.
        # If Node.start() is called, it populates self.node.device_capabilities via await device_capabilities()
        # For these tests, we'll often mock device_capabilities directly.
        pass


    @patch('exo.orchestration.node.device_capabilities') # Mocks the function call within Node class
    @patch.dict(sys.modules, {'vllm': MagicMock(), 'llama_cpp': None, 'openvino.runtime': None})
    @patch('os.getenv')
    async def test_manual_override_vllm_available(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = "vllm"
        self.node.device_capabilities = create_mock_caps(chip="NVIDIA RTX 4090", available_memory=24000) # Set manually

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, VLLMInferenceEngine)
        mock_getenv.assert_called_with("EXO_INFERENCE_ENGINE")

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': MagicMock(), 'openvino.runtime': None})
    @patch('os.getenv')
    async def test_manual_override_llamacpp_available(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = "llama_cpp"
        self.node.device_capabilities = create_mock_caps() # Generic CPU

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, LlamaCppInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': None, 'openvino.runtime': MagicMock()})
    @patch('os.getenv')
    async def test_manual_override_openvino_available(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = "openvino"
        self.node.device_capabilities = create_mock_caps(chip="Intel CPU")

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, OpenVINOInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': MagicMock(), 'openvino.runtime': None}) # llama_cpp is available for fallback
    @patch('os.getenv')
    async def test_manual_override_vllm_unavailable_falls_back_auto(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = "vllm" # Try to get vllm
        # Remove vllm from sys.modules for this test case specifically
        with patch.dict(sys.modules, {'vllm': None}):
            self.node.device_capabilities = create_mock_caps(chip="NVIDIA RTX 4090", available_memory=24000)
            await self.node.select_best_inference_engine()
            # Should fall back: given priority, LlamaCpp if NVIDIA is present and vLLM fails, then Tinygrad.
            # The current logic is: vllm -> openvino -> mlx -> llama_cpp -> tinygrad
            # So if vllm fails, and it's NVIDIA (not Intel, not Apple), it should go to llama_cpp
            self.assertIsInstance(self.node.inference_engine, LlamaCppInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': MagicMock(), 'llama_cpp': MagicMock(), 'openvino.runtime': MagicMock(), 'mlx': MagicMock()})
    @patch('os.getenv') # Ensure os.getenv is patched
    async def test_auto_select_vllm_for_nvidia_gpu(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None # No manual override
        self.node.device_capabilities = create_mock_caps(chip="NVIDIA GEFORCE RTX 4090", available_memory=10000)

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, VLLMInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': None, 'openvino.runtime': MagicMock(), 'mlx': None})
    @patch('os.getenv')
    async def test_auto_select_openvino_for_intel_cpu(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None
        self.node.device_capabilities = create_mock_caps(chip="Intel Core i9", model="Intel NUC")

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, OpenVINOInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    # For MLX, we assume it's available if on Apple Silicon. get_inference_engine('mlx',...) handles it.
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': None, 'openvino.runtime': None})
    @patch('os.getenv')
    async def test_auto_select_mlx_for_apple_silicon(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None
        self.node.device_capabilities = create_mock_caps(chip="Apple M1 Max", available_memory=32000)
        # Ensure the initial engine is not MLX to see a switch
        self.node.inference_engine = MagicMock(spec=InferenceEngine)
        self.node.inference_engine.__class__.__name__ = "SomeOtherEngine"

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, MLXDynamicShardInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': MagicMock(), 'openvino.runtime': None, 'mlx': None})
    @patch('os.getenv')
    async def test_auto_select_llamacpp_for_generic_cpu(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None
        self.node.device_capabilities = create_mock_caps(chip="Generic CPU", model="Old PC")

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, LlamaCppInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    # Simulate no specific libraries are available
    @patch.dict(sys.modules, {'vllm': None, 'llama_cpp': None, 'openvino.runtime': None, 'mlx': None})
    @patch('os.getenv')
    async def test_auto_select_tinygrad_as_fallback(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None
        self.node.device_capabilities = create_mock_caps(chip="Very Generic CPU")
        # Ensure initial engine is not Tinygrad
        self.node.inference_engine = MagicMock(spec=InferenceEngine)
        self.node.inference_engine.__class__.__name__ = "SomeOtherEngine"

        await self.node.select_best_inference_engine()
        self.assertIsInstance(self.node.inference_engine, TinygradDynamicShardInferenceEngine)

    @patch('exo.orchestration.node.device_capabilities')
    @patch('os.getenv')
    async def test_dummy_engine_skips_selection(self, mock_getenv, mock_device_capabilities_func):
        # Start with DummyInferenceEngine
        self.node.inference_engine = DummyInferenceEngine() # No shard_downloader needed for dummy's __init__
        self.node.device_capabilities = create_mock_caps(chip="NVIDIA GEFORCE RTX 4090", available_memory=10000)
        # Mock get_inference_engine to see if it's NOT called
        with patch('exo.orchestration.node.get_inference_engine') as mock_get_engine_factory:
            await self.node.select_best_inference_engine()
            self.assertIsInstance(self.node.inference_engine, DummyInferenceEngine)
            mock_get_engine_factory.assert_not_called()
            mock_getenv.assert_not_called() # EXO_INFERENCE_ENGINE should not be checked if dummy

    @patch('exo.orchestration.node.device_capabilities')
    @patch.dict(sys.modules, {'vllm': MagicMock()})
    @patch('os.getenv')
    async def test_engine_unchanged_if_already_optimal_and_no_override(self, mock_getenv, mock_device_capabilities_func):
        mock_getenv.return_value = None # No manual override
        self.node.device_capabilities = create_mock_caps(chip="NVIDIA GEFORCE RTX 4090", available_memory=10000)

        # Start with VLLM engine already set
        vllm_engine_instance = VLLMInferenceEngine(self.mock_shard_downloader)
        self.node.inference_engine = vllm_engine_instance

        # Mock the factory to see if it's called to create a *new* instance
        with patch('exo.orchestration.node.get_inference_engine') as mock_get_engine_factory:
            mock_get_engine_factory.return_value = vllm_engine_instance # If called, return same type
            await self.node.select_best_inference_engine()
            self.assertIs(self.node.inference_engine, vllm_engine_instance) # Should be the same instance
            # The factory should not be called if the class name matches and is considered optimal.
            # Current logic: if self.inference_engine.__class__.__name__.lower().replace(...) != selected_engine_name.lower():
            # So if names match, it won't call the factory.
            mock_get_engine_factory.assert_not_called()
