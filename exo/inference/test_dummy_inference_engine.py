import pytest
import numpy as np
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.inference.shard import Shard


@pytest.mark.asyncio
async def test_dummy_inference_specific():
  engine = DummyInferenceEngine()
  test_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)
  test_prompt = "This is a test prompt"

  result, _ = await engine.infer_prompt("test_request", test_shard, test_prompt)

  print(f"Inference result shape: {result.shape}")

  assert result.shape[0] == 1, "Result should be a 2D array with first dimension 1"


@pytest.mark.asyncio
async def test_dummy_inference_engine():
  # Initialize the DummyInferenceEngine
  engine = DummyInferenceEngine()

  # Create a test shard
  shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)

  # Test infer_prompt
  output, _ = await engine.infer_prompt("test_id", shard, "Test prompt")

  assert isinstance(output, np.ndarray), "Output should be a numpy array"
  assert output.ndim == 2, "Output should be 2-dimensional"

  # Test infer_tensor
  input_tensor = np.array([[1, 2, 3]])
  output, _ = await engine.infer_tensor("test_id", shard, input_tensor)

  assert isinstance(output, np.ndarray), "Output should be a numpy array"
  assert output.ndim == 2, "Output should be 2-dimensional"

  print("All tests passed!")


if __name__ == "__main__":
  import asyncio
  asyncio.run(test_dummy_inference_engine())
  asyncio.run(test_dummy_inference_specific())
