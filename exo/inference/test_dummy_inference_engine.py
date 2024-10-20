import pytest
import json
import numpy as np
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.shard import Shard

class MockShardDownloader:
    async def ensure_shard(self, shard):
        pass
@pytest.mark.asyncio
async def test_dummy_inference_specific():
    engine = DummyInferenceEngine(MockShardDownloader())
    test_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)
    test_prompt = "This is a test prompt"
    
    result, state, is_finished = await engine.infer_prompt("test_request", test_shard, test_prompt)
    
    print(f"Inference result shape: {result.shape}")
    print(f"Inference state: {state}")
    print(f"Is finished: {is_finished}")
    
    assert result.shape[0] == 1, "Result should be a 2D array with first dimension 1"
    assert isinstance(json.loads(state), dict), "State should be a valid JSON string"
    assert isinstance(is_finished, bool), "is_finished should be a boolean"

@pytest.mark.asyncio
async def test_dummy_inference_engine():
    # Initialize the DummyInferenceEngine
    engine = DummyInferenceEngine(MockShardDownloader())
    
    # Create a test shard
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)
    
    # Test infer_prompt
    output, state, is_finished = await engine.infer_prompt("test_id", shard, "Test prompt")
    
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.ndim == 2, "Output should be 2-dimensional"
    assert isinstance(state, str), "State should be a string"
    assert isinstance(is_finished, bool), "is_finished should be a boolean"

    # Test infer_tensor
    input_tensor = np.array([[1, 2, 3]])
    output, state, is_finished = await engine.infer_tensor("test_id", shard, input_tensor)
    
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.ndim == 2, "Output should be 2-dimensional"
    assert isinstance(state, str), "State should be a string"
    assert isinstance(is_finished, bool), "is_finished should be a boolean"

    print("All tests passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dummy_inference_engine())
    asyncio.run(test_dummy_inference_specific())