import io

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.worker.engines.mlx.disaggregated.adapter import (
    array_to_bytes,
    bhsd_to_nhd,
    bytes_to_array,
    chunk_to_mlx_nhd,
    inject_arrays_cache,
    inject_kv_chunk,
    nhd_to_bhsd,
    send_mlx_kv_cache,
)
from exo.worker.engines.mlx.disaggregated.protocol import (
    ArraysState,
    Done,
    KVChunk,
    TensorBlob,
    make_header,
    read_header,
    read_message,
    write_done,
    write_header,
)


def _equal(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _rand(shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    mx.random.seed(0)
    return (mx.random.uniform(shape=shape) * 10).astype(dtype)


def test_bytes_roundtrip_bf16() -> None:
    x = _rand((2, 3, 4), mx.bfloat16)
    y = bytes_to_array(array_to_bytes(x), (2, 3, 4), "bfloat16")
    assert _equal(x, y)


def test_bytes_roundtrip_f16() -> None:
    x = _rand((5,), mx.float16)
    y = bytes_to_array(array_to_bytes(x), (5,), "float16")
    assert _equal(x, y)


def test_bytes_roundtrip_f32() -> None:
    x = _rand((2, 2), mx.float32)
    y = bytes_to_array(array_to_bytes(x), (2, 2), "float32")
    assert _equal(x, y)


def test_bhsd_nhd_roundtrip() -> None:
    bhsd = _rand((1, 4, 7, 8), mx.float32)
    nhd = bhsd_to_nhd(bhsd)
    assert tuple(nhd.shape) == (7, 4, 8)
    back = nhd_to_bhsd(nhd)
    assert _equal(bhsd, back)


def test_kv_cache_inject_roundtrip() -> None:
    n_heads, seq_len, head_dim = 3, 5, 4
    k_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.float32)
    v_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.float32)
    k_nhd = bhsd_to_nhd(k_bhsd)
    v_nhd = bhsd_to_nhd(v_bhsd)

    cache = KVCache()
    inject_kv_chunk(cache, k_nhd, v_nhd, offset=seq_len)
    assert cache.offset == seq_len
    assert cache.keys is not None and cache.values is not None
    assert _equal(cache.keys, k_bhsd)
    assert _equal(cache.values, v_bhsd)


def test_arrays_cache_inject() -> None:
    a = _rand((3,), mx.float32)
    b = _rand((2, 2), mx.bfloat16)
    blobs = [
        TensorBlob(dtype="float32", shape=(3,), data=array_to_bytes(a)),
        TensorBlob(dtype="bfloat16", shape=(2, 2), data=array_to_bytes(b)),
    ]
    cache = ArraysCache(size=2)
    inject_arrays_cache(cache, blobs)
    s0 = cache.state[0]
    s1 = cache.state[1]
    assert s0 is not None and s1 is not None
    assert _equal(s0, a)
    assert _equal(s1, b)


def test_send_mlx_cache_end_to_end() -> None:
    n_heads, head_dim = 2, 4
    seq_len = 3

    k_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)
    v_bhsd = _rand((1, n_heads, seq_len, head_dim), mx.bfloat16)

    src = KVCache()
    src.keys = k_bhsd
    src.values = v_bhsd
    src.offset = seq_len

    buf = io.BytesIO()
    hdr = make_header(num_layers=1, dtype="bfloat16")
    write_header(buf, hdr)
    tokens = send_mlx_kv_cache(buf, [src])
    write_done(buf, tokens)
    buf.seek(0)

    got_hdr = read_header(buf)
    assert got_hdr["num_layers"] == 1

    msg = read_message(buf, got_hdr)
    assert isinstance(msg, KVChunk)
    k_nhd, v_nhd = chunk_to_mlx_nhd(msg)
    dst = KVCache()
    inject_kv_chunk(dst, k_nhd, v_nhd, offset=msg.num_tokens)

    done = read_message(buf, got_hdr)
    assert isinstance(done, Done)
    assert done.total_tokens == seq_len

    assert dst.offset == seq_len
    assert dst.keys is not None and dst.values is not None
    assert _equal(dst.keys, k_bhsd)
    assert _equal(dst.values, v_bhsd)
    _ = ArraysState
