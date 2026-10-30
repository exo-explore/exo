"""Copies of mlx arrays and caches that don't keep the original's buffers alive."""

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import RotatingKVCache


def detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = detached_copy(cache.keys[..., -n:, :])
    v_slice = detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap
