"""Tests for PyTorch engine pipeline sharding.

NOTE: These tests require downloading models and are marked as slow.
Run with: pytest -m slow tests/worker/engines/test_pytorch_engine.py
"""

import pytest

# Skip these tests by default as they require model downloads
pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_pytorch_engine_pipeline_sharding():
    """Test that pipeline sharding correctly slices model layers.

    This test downloads TinyLlama and verifies that the auto_parallel
    function correctly slices the model's transformer layers.
    """
    # Import here to avoid import errors if torch/transformers not installed
    from transformers import AutoModelForCausalLM

    from exo.worker.engines.pytorch.auto_parallel import (
        MockDistributedGroup,
        PipelineShardMetadata,
        _get_layers,
        pipeline_auto_parallel,
    )

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    total_model_layers = 22  # TinyLlama has 22 transformer layers

    # Load the full model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Verify we can find the layers
    layers, attr_path = _get_layers(model)
    assert len(layers) == total_model_layers, (
        f"Expected {total_model_layers} layers, got {len(layers)}"
    )

    # Simulate a shard for device 0 in a 2-device world
    world_size = 2
    device_rank = 0
    layers_per_device = total_model_layers // world_size
    start_layer = device_rank * layers_per_device
    end_layer = start_layer + layers_per_device
    expected_layers_in_shard = end_layer - start_layer

    shard_metadata = PipelineShardMetadata(
        start_layer=start_layer,
        end_layer=end_layer,
        device_rank=device_rank,
        world_size=world_size,
    )

    # Apply pipeline parallelism
    group = MockDistributedGroup()
    model = pipeline_auto_parallel(model, group, shard_metadata)

    # Verify the model now has only the sliced layers
    sliced_layers, _ = _get_layers(model)
    actual_layers = len(sliced_layers)

    assert actual_layers == expected_layers_in_shard, (
        f"Expected {expected_layers_in_shard} layers, but got {actual_layers}."
    )


@pytest.mark.asyncio
async def test_get_layers_detects_architecture():
    """Test that _get_layers correctly detects different model architectures."""
    import torch.nn as nn

    from exo.worker.engines.pytorch.auto_parallel import _get_layers

    # Create a mock Llama-style model
    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])

    mock_model = MockLlamaModel()
    layers, attr_path = _get_layers(mock_model)

    assert len(layers) == 4
    assert attr_path == ["model", "layers"]


@pytest.mark.asyncio
async def test_get_layers_raises_for_unknown_architecture():
    """Test that _get_layers raises a helpful error for unknown models."""
    import torch.nn as nn

    from exo.worker.engines.pytorch.auto_parallel import _get_layers

    class UnknownModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weird_structure = nn.Linear(10, 10)

    with pytest.raises(AttributeError) as exc_info:
        _get_layers(UnknownModel())

    assert "Supported architectures" in str(exc_info.value)
    assert "UnknownModel" in str(exc_info.value)
