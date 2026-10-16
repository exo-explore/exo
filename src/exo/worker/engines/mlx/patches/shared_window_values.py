from collections.abc import Callable
from functools import partial
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

_Fetched = tuple[mx.array, mx.array]

_original_rotating_update = cast(
    Callable[[RotatingKVCache, mx.array, mx.array], _Fetched],
    RotatingKVCache.update_and_fetch,
)
_original_batch_rotating_update = cast(
    Callable[[BatchRotatingKVCache, mx.array, mx.array], _Fetched],
    BatchRotatingKVCache.update_and_fetch,
)


def _update_sharing_values(
    cache: RotatingKVCache | BatchRotatingKVCache,
    keys: mx.array,
    values: mx.array,
    original: Callable[[mx.array, mx.array], _Fetched],
) -> _Fetched:
    """Keep the values as the keys when both are the same array.

    DeepSeek V4 attention updates its sliding window with
    ``update_and_fetch(kv, kv)`` and only reads the keys. Upstream still
    writes the values in place, and as nothing evaluates them, every decode
    step chains another unevaluated slice update that holds that step's kv.
    On 4x M5 Pro a long agent session built this up to hundreds of thousands
    of live Metal buffers (each at least a page) until Metal aborted with
    "Resource limit (499000) exceeded".
    """
    if keys is not values:
        return original(keys, values)
    if cache.values is cache.keys and cache.keys is not None:
        # A separate handle, so the in-place write lands on the keys once.
        cache.values = mx.array(cache.keys)
    fetched, _ = original(keys, values)
    cache.values = cache.keys
    return fetched, fetched


def _rotating_update(
    self: RotatingKVCache, keys: mx.array, values: mx.array
) -> _Fetched:
    return _update_sharing_values(
        self, keys, values, partial(_original_rotating_update, self)
    )


def _batch_rotating_update(
    self: BatchRotatingKVCache, keys: mx.array, values: mx.array
) -> _Fetched:
    return _update_sharing_values(
        self, keys, values, partial(_original_batch_rotating_update, self)
    )


def patch_shared_window_values() -> None:
    RotatingKVCache.update_and_fetch = _rotating_update
    BatchRotatingKVCache.update_and_fetch = _batch_rotating_update
