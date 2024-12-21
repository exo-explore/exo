import pytest
import numpy as np
from exo.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine
from exo.inference.shard import Shard

class MockShardDownloader:
    async def ensure_shard(self, shard):
        pass

@pytest.mark.asyncio
async def test_llama_cpp_inference_engine():
    engine = LlamaCppInferenceEngine(MockShardDownloader())
    test_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)
    test_prompt = "This is a test prompt"

    # Test encode
    encoded = await engine.encode(test_shard, test_prompt)
    assert isinstance(encoded, np.ndarray), "Encoded output should be a numpy array"

    # Test sample
    sampled = await engine.sample(encoded)
    assert isinstance(sampled, np.ndarray), "Sampled output should be a numpy array"

    # Test decode
    decoded = await engine.decode(test_shard, sampled)
    assert isinstance(decoded, str), "Decoded output should be a string"

    # Test infer_tensor
    input_data = np.array([[1, 2, 3]])
    inferred = await engine.infer_tensor("test_request", test_shard, input_data)
    assert isinstance(inferred, np.ndarray), "Inferred output should be a numpy array"

    # Test load_checkpoint
    await engine.load_checkpoint(test_shard, "path/to/checkpoint")

    # Test save_checkpoint
    await engine.save_checkpoint(test_shard, "path/to/checkpoint")

    # Test ensure_shard
    await engine.ensure_shard(test_shard)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llama_cpp_inference_engine())
