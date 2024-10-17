import pytest
import asyncio
import numpy as np
import json
from exo.inference.shard import Shard
from exo.inference.DummyInferenceEngine2 import DummyInferenceEngine2 

@pytest.fixture
def dummy_engine():
    # Create a DummyInferenceEngine2 instance
    shard_downloader = None  # We don't need a real shard downloader for this test
    return DummyInferenceEngine2(shard_downloader)

@pytest.fixture
def shard():
    # Create a dummy shard
    return Shard(model_id="dummy_model", start_layer=0, end_layer=10, n_layers=10)

@pytest.mark.asyncio
async def test_infer_prompt(dummy_engine, shard):
    # Test inference on a simple prompt
    prompt = "Test prompt"
    
    # Perform inference
    output, state, is_finished = await dummy_engine.infer_prompt("test_id", shard, prompt)

    # Check output is a numpy array
    assert isinstance(output, np.ndarray), "Output should be a numpy array."
    
    # Check state is a valid JSON string
    state_dict = json.loads(state)
    assert isinstance(state_dict, dict), "State should be a valid JSON dict."

    # Check the state update and progress
    assert "start_pos" in state_dict, "State should contain 'start_pos'."
    
    # Account for possible off-by-one errors due to tokenization/inference
    expected_start_pos = len(prompt)  # Check length matches prompt's tokens
    assert state_dict["start_pos"] == expected_start_pos, f"Expected start_pos {expected_start_pos}, but got {state_dict['start_pos']}."

    # Check finished flag
    assert isinstance(is_finished, bool), "is_finished should be a boolean."


@pytest.mark.asyncio
async def test_infer_tensor(dummy_engine, shard):
    # Test inference with tensor input
    input_tensor = np.random.randn(1, 10)  # Random input tensor

    # Perform inference
    output, state, is_finished = await dummy_engine.infer_tensor("test_id", shard, input_tensor)

    # Check output is a numpy array
    assert isinstance(output, np.ndarray), "Output should be a numpy array."

    # Check state is a valid JSON string
    state_dict = json.loads(state)
    assert isinstance(state_dict, dict), "State should be a valid JSON dict."

    # Check start position update based on input tensor shape
    assert "start_pos" in state_dict, "State should contain 'start_pos'."
    assert state_dict["start_pos"] == input_tensor.shape[1], "Start position should be updated correctly."

@pytest.mark.asyncio
async def test_shard_loading(dummy_engine, shard):
    # Simulate shard loading
    await dummy_engine.ensure_shard(shard)

    # Check that the shard has been loaded
    assert dummy_engine.shard == shard, "Shard should be loaded into the engine."

    # Test that calling ensure_shard with the same shard doesn't re-trigger shard loading
    await dummy_engine.ensure_shard(shard)
    # No exception means the shard handling works fine

@pytest.mark.asyncio
async def test_inference_state_handling(dummy_engine, shard):
    # Simulate initial inference state
    initial_state = json.dumps({
        "start_pos": 0,
        "n_captured_toks": 0
    })

    prompt = "This is a test"
    
    # Perform inference
    output, state, _ = await dummy_engine.infer_prompt("test_id", shard, prompt, inference_state=initial_state)

    # Check that state is updated correctly
    state_dict = json.loads(state)
    assert state_dict["start_pos"] == len(prompt), "Start position should be updated based on the prompt length."
    assert state_dict["n_captured_toks"] == len(prompt), "n_captured_toks should match the prompt length."

@pytest.mark.asyncio
async def test_random_finish_behavior(dummy_engine, shard):
    prompt = "Finish test"

    finished_count = 0
    trials = 100

    for _ in range(trials):
        _, _, is_finished = await dummy_engine.infer_prompt("test_id", shard, prompt)
        if is_finished:
            finished_count += 1

    # Since the finish behavior has a 20% chance, we check if the observed finish count is in a reasonable range
    assert 10 <= finished_count <= 20, "is_finished should be True about 20% of the time."
