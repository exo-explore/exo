from typing import Any

import torch
import torch.nn as nn

from exo.shared.types.worker.shards import PipelineShardMetadata


# Placeholder for distributed group (will be replaced by torch.distributed.Group)
class MockDistributedGroup:
    pass


# Placeholder for CustomMlxLayer equivalent
class CustomPytorchLayer(nn.Module):
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.original_layer = original_layer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.original_layer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # First, let nn.Module handle its standard attribute lookup (_modules, _parameters, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # If not found via nn.Module, delegate to original_layer for model-specific attributes
        # (e.g., 'attention_type' for Qwen2)
        # Access _modules directly to avoid recursion
        _modules: dict[str, nn.Module] = object.__getattribute__(self, "_modules")
        if "original_layer" in _modules:
            return getattr(_modules["original_layer"], name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class PipelineFirstLayer(CustomPytorchLayer):
    """Wrapper for the first layer in a pipeline-parallel shard.

    TODO: Implement inter-device communication via torch.distributed.
    Currently a no-op pass-through — distributed send/recv not yet implemented.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        r: int,  # device_rank
        group: MockDistributedGroup,
    ):
        super().__init__(original_layer)
        self.r = r
        self.group = group

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # In a real distributed setup, this would receive from the previous rank
        # For now, we just pass through
        # if self.r != 0:
        #     x = receive_from_previous_rank(x, self.group)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomPytorchLayer):
    """Wrapper for the last layer in a pipeline-parallel shard.

    TODO: Implement inter-device communication via torch.distributed.
    Currently a no-op pass-through — distributed send/recv not yet implemented.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        r: int,  # device_rank
        s: int,  # world_size
        group: MockDistributedGroup,
    ):
        super().__init__(original_layer)
        self.r = r
        self.s = s
        self.group = group

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        output: torch.Tensor = self.original_layer(x, *args, **kwargs)
        # In a real distributed setup, this would send to the next rank
        # For now, we just pass through
        # if self.r != self.s - 1:
        #     send_to_next_rank(output, self.group)
        return output


# Supported model architecture patterns for layer detection
# Format: (attr_path_to_layers, description)
# Add new architectures here as they're validated
SUPPORTED_LAYER_PATTERNS: list[tuple[list[str], str]] = [
    # Llama, Mistral, TinyLlama, etc.
    (["model", "layers"], "Llama-style (model.model.layers)"),
    # Some wrapped models
    (["base_model", "layers"], "base_model.layers"),
    # GPT-2, GPT-Neo
    (["transformer", "h"], "GPT-2 style (model.transformer.h)"),
    # GPT-J, GPT-NeoX
    (["gpt_neox", "layers"], "GPT-NeoX style (model.gpt_neox.layers)"),
    # Falcon
    (["transformer", "h"], "Falcon style (model.transformer.h)"),
    # BLOOM
    (["transformer", "h"], "BLOOM style (model.transformer.h)"),
    # OPT
    (["model", "decoder", "layers"], "OPT style (model.model.decoder.layers)"),
]


def _get_nested_attr(obj: object, attr_path: list[str]) -> nn.ModuleList | None:
    """Safely traverse a nested attribute path."""
    current = obj
    for attr in attr_path:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    if isinstance(current, nn.ModuleList):
        return current
    return None


def _set_nested_attr(obj: object, attr_path: list[str], value: nn.ModuleList) -> bool:
    """Safely set a nested attribute."""
    if len(attr_path) == 0:
        return False
    current = obj
    for attr in attr_path[:-1]:
        if not hasattr(current, attr):
            return False
        current = getattr(current, attr)
    setattr(current, attr_path[-1], value)
    return True


def _get_layers(model: nn.Module) -> tuple[nn.ModuleList, list[str]]:
    """
    Find the transformer layers in a model using known architecture patterns.

    Returns:
        Tuple of (layers ModuleList, attribute path used to find them)

    Raises:
        AttributeError: If no supported layer pattern is found
    """
    for attr_path, _description in SUPPORTED_LAYER_PATTERNS:
        layers = _get_nested_attr(model, attr_path)
        if layers is not None:
            return layers, attr_path

    # Build helpful error message
    supported = "\n".join(f"  - {desc}" for _, desc in SUPPORTED_LAYER_PATTERNS)
    raise AttributeError(
        f"Could not find transformer layers in model. "
        f"Supported architectures:\n{supported}\n"
        f"Model type: {type(model).__name__}"
    )


def _set_layers(
    model: nn.Module, new_layers: nn.ModuleList, attr_path: list[str]
) -> None:
    """Set the transformer layers using the previously discovered attribute path."""
    if not _set_nested_attr(model, attr_path, new_layers):
        raise AttributeError(f"Failed to set layers at path: {'.'.join(attr_path)}")


def pipeline_auto_parallel(
    model: nn.Module,
    group: MockDistributedGroup,  # This will be torch.distributed.Group
    model_shard_meta: PipelineShardMetadata,
) -> nn.Module:
    """
    Automatically parallelize a PyTorch model across multiple devices using pipeline parallelism.
    Args:
        model: The model to parallelize (must have a 'layers' or 'h' property).
        group: The distributed group for communication.
        model_shard_meta: The metadata for the model shard.
    Returns:
        The parallelized model with only the assigned layers.
    """
    all_layers, attr_path = _get_layers(model)

    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    # Slice the layers
    sliced_layers = list(all_layers[start_layer:end_layer])

    # Wrap the first and last layers for inter-device communication
    # (Communication logic will be added later)
    if sliced_layers:  # Ensure there are layers to wrap
        sliced_layers[0] = PipelineFirstLayer(
            sliced_layers[0], device_rank, group=group
        )
        sliced_layers[-1] = PipelineLastLayer(
            sliced_layers[-1], device_rank, world_size, group=group
        )

    # Replace the model's layers with the sliced ones
    _set_layers(model, nn.ModuleList(sliced_layers), attr_path)

    return model
