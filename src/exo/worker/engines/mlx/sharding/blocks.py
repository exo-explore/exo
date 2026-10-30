from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients


class LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: LayerCallable):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)  # pyright: ignore[reportUnknownMemberType]

    @property
    def original_layer(self) -> LayerCallable:
        return cast(LayerCallable, self["_original_layer"])

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = cast(LayerCallable, self["_original_layer"])
                return getattr(original_layer, name)


class ShardedMoE(CustomMlxLayer):
    """Wraps any MoE layer with distributed sum_gradients / all_sum."""

    def __init__(self, layer: LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
