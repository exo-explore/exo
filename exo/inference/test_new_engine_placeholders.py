import unittest
from unittest.mock import MagicMock

from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.inference.llama_cpp.llama_cpp_inference_engine import LlamaCppInferenceEngine
from exo.inference.vllm.vllm_inference_engine import VLLMInferenceEngine
from exo.inference.openvino.openvino_inference_engine import OpenVINOInferenceEngine
from exo.download.shard_download import ShardDownloader


class TestNewEnginePlaceholders(unittest.TestCase):

    def setUp(self):
        self.mock_shard_downloader = MagicMock(spec=ShardDownloader)

    def test_can_get_llama_cpp_engine_placeholder(self):
        try:
            engine = get_inference_engine("llama_cpp", self.mock_shard_downloader)
            self.assertIsInstance(engine, LlamaCppInferenceEngine)
            self.assertIsInstance(engine, InferenceEngine)
        except Exception as e:
            self.fail(f"Error getting 'llama_cpp': {e}")

    def test_can_get_vllm_engine_placeholder(self):
        try:
            engine = get_inference_engine("vllm", self.mock_shard_downloader)
            self.assertIsInstance(engine, VLLMInferenceEngine)
            self.assertIsInstance(engine, InferenceEngine)
        except Exception as e:
            self.fail(f"Error getting 'vllm': {e}")

    def test_can_get_openvino_engine_placeholder(self):
        try:
            engine = get_inference_engine("openvino", self.mock_shard_downloader)
            self.assertIsInstance(engine, OpenVINOInferenceEngine)
            self.assertIsInstance(engine, InferenceEngine)
        except Exception as e:
            self.fail(f"Error getting 'openvino': {e}")

    def test_get_unknown_engine_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_inference_engine("unknown_engine_xyz", self.mock_shard_downloader)

if __name__ == '__main__':
    unittest.main()
