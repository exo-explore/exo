from typing import cast, override, Protocol, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn  # pyright: ignore[reportMissingTypeStubs]
from exo.shared.types.worker.shards import PipelineShardMetadata


class IdentityLayer(nn.Module):
    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        # Set twice to avoid __setattr__ recursion
        object.__setattr__(self, "_original_layer", original_layer)
        self.original_layer: _LayerCallable = original_layer

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = object.__getattribute__(self, "_original_layer")
                return object.__getattribute__(original_layer, name)


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(self, original_layer: _LayerCallable, r: int, s: int):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s

    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            x = mx.distributed.recv_like(x, (self.r - 1))
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(self, original_layer: _LayerCallable, r: int, s: int):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s

    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        output: mx.array = self.original_layer(x, *args, **kwargs)
        if self.r != self.s - 1:
            output = mx.distributed.send(output, (self.r + 1) % self.s)
        output = mx.distributed.all_gather(output)[-output.shape[0] :]
        return output


def inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    raise ValueError("Model must either have a 'model' or 'transformer' attribute")


# def auto_parallel(model: nn.Module, rank: int, size: int, start_layer: int, end_layer: int) -> nn.Module:
def auto_parallel(
    model: nn.Module, model_shard_meta: PipelineShardMetadata
) -> nn.Module:
    """
    Automatically parallelize a model across multiple devices.

    Args:
      model: The model to parallelize (must have a 'layers' or 'h' property)
      model_shard_meta: The metadata for the model shard

    Returns:
      The parallelized model
    """
    inner_model_instance: nn.Module = inner_model(model)

    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    else:
        layers = cast(list[_LayerCallable], inner_model_instance.h)

    layers[: model_shard_meta.start_layer] = [
        IdentityLayer() for _ in range(model_shard_meta.start_layer)
    ]
    layers[model_shard_meta.end_layer :] = [
        IdentityLayer() for _ in range(len(layers) - model_shard_meta.end_layer)
    ]
    layers[model_shard_meta.start_layer] = PipelineFirstLayer(
        layers[model_shard_meta.start_layer],
        model_shard_meta.device_rank,
        model_shard_meta.world_size,
    )
    layers[model_shard_meta.end_layer - 1] = PipelineLastLayer(
        layers[model_shard_meta.end_layer - 1],
        model_shard_meta.device_rank,
        model_shard_meta.world_size,
    )

    # At this point `layers` *must* be a concrete list.
    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return model
