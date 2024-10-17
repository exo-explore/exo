import asyncio
import pytest
import numpy as np
from exo.inference.DummyInferenceEngine import DummyInferenceEngine
from exo.inference.shard import Shard

@pytest.mark.asyncio
async def test_dummy_engine_infer_prompt():
    # Instantiate DummyInferenceEngine with random output and specific shape
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(1, 32), latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)

    # Perform inference on prompt
    output, state, is_finished = await dummy_engine.infer_prompt("test_id", shard, "Test prompt")
    
    # Assertions
    assert isinstance(output, np.ndarray), "Output should be a numpy array."
    assert output.shape == (1, 32), "Output should have the specified shape."
    assert isinstance(state, str), "State should be a string."
    assert isinstance(is_finished, bool), "is_finished should be a boolean."

@pytest.mark.asyncio
async def test_dummy_engine_infer_tensor():
    # Instantiate DummyInferenceEngine with random output and specific shape
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(1, 32), latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)
    
    # Random input tensor for the inference
    input_tensor = np.random.randn(1, 16)
    
    # Perform inference on tensor
    output, state, is_finished = await dummy_engine.infer_tensor("test_id", shard, input_tensor)
    
    # Assertions
    assert isinstance(output, np.ndarray), "Output should be a numpy array."
    assert output.shape == (1, 32), "Output should have the specified shape."
    assert isinstance(state, str), "State should be a string."
    assert isinstance(is_finished, bool), "is_finished should be a boolean."

@pytest.mark.asyncio
async def test_dummy_engine_static_output():
    # Static output values to compare against
    static_value = np.array([1, 2, 3, 4])
    # Instantiate DummyInferenceEngine with static output type
    dummy_engine = DummyInferenceEngine(output_type="static", output_value=static_value, latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)

    # Perform inference on prompt
    output, _, _ = await dummy_engine.infer_prompt("test_id", shard, "Test prompt")
    
    # Validate that the output matches the static value
    np.testing.assert_array_equal(output, static_value), "Static output should match the provided value."

@pytest.mark.asyncio
async def test_dummy_engine_latency():
    # Instantiate DummyInferenceEngine with static output and latency
    dummy_engine = DummyInferenceEngine(output_type="static", output_value=np.array([1]), latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)

    # Measure latency of the inference call
    start_time = asyncio.get_event_loop().time()
    await dummy_engine.infer_prompt("test_id", shard, "Test prompt")
    elapsed_time = asyncio.get_event_loop().time() - start_time
    
    # Assert that latency falls within expected bounds
    assert 0.05 <= elapsed_time <= 0.15, f"Expected latency to be around 0.1s, but got {elapsed_time}s."

@pytest.mark.asyncio
async def test_dummy_engine_error_handling():
    # Ensure that creating a DummyInferenceEngine with invalid parameters raises an error
    with pytest.raises(ValueError):
        DummyInferenceEngine(output_type="static", output_value=None)

@pytest.mark.asyncio
async def test_dummy_engine_multiple_inferences():
    # Instantiate DummyInferenceEngine with random output and shape
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(1, 32), latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)
    
    # Perform multiple inferences and track results
    results = []
    for _ in range(5):
        output, _, is_finished = await dummy_engine.infer_prompt("test_id", shard, "Test prompt")
        results.append((output, is_finished))
    
    # Assert that there are 5 results and some are marked as finished
    assert len(results) == 5, "Should have 5 inference results."
    assert any(is_finished for _, is_finished in results), "At least one inference should be marked as finished."
    assert not all(is_finished for _, is_finished in results), "Not all inferences should be marked as finished."

@pytest.mark.asyncio
async def test_dummy_engine_output_shape_consistency():
    # Instantiate DummyInferenceEngine with random output and specific shape
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(2, 3, 4), latency_mean=0.1, latency_stddev=0.01)
    # Adjust Shard instantiation according to the working model
    shard = Shard(model_id="dummy", start_layer=0, end_layer=1, n_layers=2)
    
    # Perform multiple inferences and ensure the output shape is consistent
    for _ in range(5):
        output, _, _ = await dummy_engine.infer_prompt("test_id", shard, "Test prompt")
        assert output.shape == (2, 3, 4), "Output shape should be consistent across multiple inferences."
