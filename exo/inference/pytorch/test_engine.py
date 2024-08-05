import unittest
from .inference import PyTorchDynamicShardInferenceEngine
from exo.inference.shard import Shard
import asyncio

class TestPyTorchDynamicShardInferenceEngine(unittest.TestCase):
    def test_one(self):
        shard = Shard(model_id="mock_model", start_layer=0, end_layer=1, n_layers=2)
        engine = PyTorchDynamicShardInferenceEngine()
        prompt_resp = asyncio.run(
            engine.infer_prompt(
                "", 
                shard, 
                "Why is the sky blue?")
        )

        self.assertIsNotNone(prompt_resp)
