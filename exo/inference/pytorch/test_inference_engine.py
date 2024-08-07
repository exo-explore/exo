import unittest
import asyncio
from exo.inference.shard import Shard
from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine

class TestPyTorchDynamicShardInferenceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Create a shard
        cls.shard = Shard(
            model_id="llama3-8b-sfr",
            start_layer=0,
            end_layer=0,
            n_layers=12
        )

        # Initialize the inference engine
        cls.engine = PyTorchDynamicShardInferenceEngine(debug=True)

    def test_infer_prompt(self):
        # Prepare the prompt
        prompt = "Why is the sky blue?"

        # Run inference
        loop = asyncio.get_event_loop()
        output_data, new_inference_state, is_eos = loop.run_until_complete(
            self.engine.infer_prompt(
                request_id="test_request", shard=self.shard, prompt=prompt
            )
        )

        # Assertions
        self.assertIsNotNone(output_data)
        self.assertIsNotNone(new_inference_state)
        self.assertFalse(is_eos)

if __name__ == '__main__':
    unittest.main()
