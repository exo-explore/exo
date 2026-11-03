import io

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)

_SEQUENCE_LENGTH = 7


def _equal(a: mx.array | None, b: mx.array | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _kv_layer(seed: int) -> KVCache:
    mx.random.seed(seed)
    cache = KVCache()
    shape = (1, 2, _SEQUENCE_LENGTH, 4)
    cache.keys = mx.random.normal(shape).astype(mx.bfloat16)
    cache.values = mx.random.normal(shape).astype(mx.bfloat16)
    cache.offset = _SEQUENCE_LENGTH
    return cache


def _arrays_layer(seed: int) -> ArraysCache:
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    cache.cache = [mx.random.normal((1, 3)), mx.random.normal((1, 2, 2))]
    return cache


def _six_layer_model_cache() -> list[KVCache | ArraysCache]:
    # Layer 4 is a state layer, as in hybrid models.
    return [
        _kv_layer(0),
        _kv_layer(1),
        _kv_layer(2),
        _kv_layer(3),
        _arrays_layer(4),
        _kv_layer(5),
    ]


def _through_wire(caches: list[KVCache | ArraysCache]) -> PrefillResult:
    stream = io.BytesIO()
    write_cache_to_wire(stream, caches, request_id="r", model_id="m")
    stream.seek(0)
    result = PrefillResult(header=read_header(stream))
    while True:
        message = read_message(stream)
        if message is None or isinstance(message, Done):
            break
        if isinstance(message, KVChunk):
            result.kv_chunks.setdefault(message.layer_idx, []).append(message)
        elif isinstance(message, ArraysState):
            result.arrays[message.layer_idx] = message.arrays
    return result


def _same_layer(a: KVCache | ArraysCache, b: KVCache | ArraysCache) -> bool:
    if isinstance(a, KVCache) and isinstance(b, KVCache):
        return (
            a.offset == b.offset
            and _equal(a.keys, b.keys)
            and _equal(a.values, b.values)
        )
    if isinstance(a, ArraysCache) and isinstance(b, ArraysCache):
        return all(_equal(x, y) for x, y in zip(a.cache, b.cache, strict=True))
    return False


def _second_pipeline_rank() -> list[KVCache | ArraysCache]:
    # A pipeline rank holding global layers 3, 4 and 5.
    return [KVCache(), ArraysCache(size=2), KVCache()]


def test_pipeline_rank_ingests_its_own_layers() -> None:
    full_model = _six_layer_model_cache()
    result = _through_wire(full_model)

    rank = _second_pipeline_rank()
    final_offset = ingest_into_mlx_cache(result, list(rank), layer_offset=3)

    assert final_offset == _SEQUENCE_LENGTH
    assert all(
        _same_layer(local, full_model[3 + index]) for index, local in enumerate(rank)
    )


def test_pipeline_rank_without_offset_gets_the_first_layers() -> None:
    # Without the offset a pipeline rank silently loads layers 0..2: no error,
    # wrong attention. This is what layer_offset fixes.
    full_model = _six_layer_model_cache()
    result = _through_wire(full_model)

    rank = _second_pipeline_rank()
    ingest_into_mlx_cache(result, list(rank))

    assert _same_layer(rank[0], full_model[0])
    assert not _same_layer(rank[0], full_model[3])
    assert not _same_layer(rank[2], full_model[5])


def test_first_rank_and_single_node_are_unchanged() -> None:
    full_model = _six_layer_model_cache()
    result = _through_wire(full_model)

    first_rank: list[KVCache | ArraysCache] = [KVCache(), KVCache(), KVCache()]
    ingest_into_mlx_cache(result, list(first_rank), layer_offset=0)

    assert all(
        _same_layer(local, full_model[index]) for index, local in enumerate(first_rank)
    )
