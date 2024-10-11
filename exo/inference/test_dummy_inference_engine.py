import asyncio
import pytest
from exo.inference.DummyInferenceEngine import DummyInferenceEngine  # Adjust the path to where you implemented DummyInferenceEngine

@pytest.mark.asyncio
async def test_dummy_inference_engine_static():
    # Test with static output
    dummy_engine = DummyInferenceEngine(output_type="static", output_value=[1, 2, 3], latency_mean=0.2, latency_stddev=0.05)
    await dummy_engine.run_inference()  # Simulate inference, check for errors
    assert dummy_engine.output_value == [1, 2, 3], "The static output should match the provided value."


@pytest.mark.asyncio
async def test_dummy_inference_engine_random():
    # Test with random output
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(128, 128), latency_mean=0.1, latency_stddev=0.01)
    await dummy_engine.run_inference()  # Simulate inference, check for errors
    output = dummy_engine.output_value
    
    # Check that the output is a list and has the correct shape
    assert isinstance(output, list), "Output should be a list."
    assert len(output) == 128, "Output should have the specified outer shape."

    # Check each sub-list for the correct length
    for sublist in output:
        assert isinstance(sublist, list), "Each output item should be a list."
        assert len(sublist) == 128, "Each output sub-list should have the specified inner shape."

    # Optionally check the type of each element
    for sublist in output:
        for element in sublist:
            assert isinstance(element, (float, int)), "Each element should be a float or int."



@pytest.mark.asyncio
async def test_dummy_inference_engine_latency():
    # Test that latency is within expected range
    dummy_engine = DummyInferenceEngine(output_type="static", output_value=[1], latency_mean=0.1, latency_stddev=0.02)
    start_time = asyncio.get_event_loop().time()
    await dummy_engine.run_inference()
    elapsed_time = asyncio.get_event_loop().time() - start_time
    assert 0.06 <= elapsed_time <= 0.14, f"Expected latency to be around 0.1s, but got {elapsed_time}s."

# run tests 
# pytest test_dummy_inference_engine.py