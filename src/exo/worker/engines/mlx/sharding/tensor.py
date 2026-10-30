from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from inspect import signature

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.runner_response import ModelLoadingResponse


def patch_tensor_model[T](model: T) -> T:
    """Patch model's __call__ to ensure distributed ops sync during inference."""
    cls = model.__class__
    original_call = cls.__call__
    call_signature = signature(original_call)

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # pyright: ignore[reportAny]
        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None and len(cache) > 0:  # pyright: ignore[reportAny]
            last = cache[-1]  # pyright: ignore[reportAny]
            dep_cache = last[0] if hasattr(last, "caches") else last  # pyright: ignore[reportAny]
            if hasattr(dep_cache, "keys"):  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # pyright: ignore[reportAny]

        return logits

    cls.__call__ = patched_call
    return model


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]: ...
