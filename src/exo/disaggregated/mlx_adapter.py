import io
from collections.abc import Sequence
from typing import BinaryIO

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from exo.disaggregated.protocol import (
    DType,
    KVChunk,
    TensorBlob,
    make_header,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)

AnyMLXCache = (
    KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList | None
)

_STR_TO_MX: dict[str, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}

_MX_TO_STR: dict[mx.Dtype, str] = {v: k for k, v in _STR_TO_MX.items()}


def mx_dtype_to_str(dtype: mx.Dtype) -> DType:
    if dtype not in _MX_TO_STR:
        raise ValueError(f"Unsupported mlx dtype on wire: {dtype}")
    return _MX_TO_STR[dtype]


def str_to_mx_dtype(dtype: DType) -> mx.Dtype:
    if dtype not in _STR_TO_MX:
        raise ValueError(f"Unsupported wire dtype: {dtype!r}")
    return _STR_TO_MX[dtype]


def array_to_bytes(t: mx.array) -> bytes:
    # bf16 has no native numpy dtype; bitcast through uint16.
    if t.dtype == mx.bfloat16:
        return np.asarray(t.view(mx.uint16)).tobytes()
    if t.dtype in (mx.float16, mx.float32):
        return np.asarray(t).tobytes()
    raise ValueError(f"Unsupported mlx dtype for wire: {t.dtype}")


def bytes_to_array(data: bytes, shape: tuple[int, ...], dtype: DType) -> mx.array:
    if dtype == "bfloat16":
        arr = np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()
        return mx.array(arr).view(mx.bfloat16)
    if dtype == "float16":
        arr = np.frombuffer(data, dtype=np.float16).reshape(shape).copy()
        return mx.array(arr)
    if dtype == "float32":
        arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
        return mx.array(arr)
    raise ValueError(f"Unsupported wire dtype for mlx: {dtype!r}")


def bhsd_to_nhd(t: mx.array) -> mx.array:
    if t.ndim != 4 or int(t.shape[0]) != 1:
        raise ValueError(f"Expected BHSD with B=1, got shape={tuple(t.shape)}")
    return mx.transpose(t[0], (1, 0, 2))


def nhd_to_bhsd(t: mx.array) -> mx.array:
    if t.ndim != 3:
        raise ValueError(f"Expected NHD (3D), got shape={tuple(t.shape)}")
    return mx.expand_dims(mx.transpose(t, (1, 0, 2)), 0)


def send_mlx_kv_cache(
    stream: BinaryIO,
    caches: list[AnyMLXCache] | Sequence[AnyMLXCache],
    *,
    max_tokens: int | None = None,
) -> int:
    tokens_sent = 0
    for layer_idx, c in enumerate(caches):
        if c is None:
            continue
        if isinstance(c, (QuantizedKVCache, CacheList)):
            continue
        if isinstance(c, (KVCache, RotatingKVCache)):
            keys = c.keys
            values = c.values
            if keys is None or values is None:
                continue
            offset = int(c.offset)
            if max_tokens is not None:
                offset = min(offset, max_tokens)
            if offset <= 0:
                continue
            # Materialize on CPU so array_to_bytes works regardless of caller's stream.
            with mx.stream(mx.Device(mx.cpu)):
                k = mx.array(keys[:, :, :offset, :])
                v = mx.array(values[:, :, :offset, :])
                k_nhd = bhsd_to_nhd(k)
                v_nhd = bhsd_to_nhd(v)
                mx.eval(k_nhd, v_nhd)
            num_tokens = int(k_nhd.shape[0])
            n_heads = int(k_nhd.shape[1])
            head_dim = int(k_nhd.shape[2])
            write_kv_chunk(
                stream,
                layer_idx=layer_idx,
                num_tokens=num_tokens,
                n_heads=n_heads,
                head_dim=head_dim,
                keys=array_to_bytes(k_nhd),
                values=array_to_bytes(v_nhd),
            )
            tokens_sent = max(tokens_sent, num_tokens)
        else:
            blobs: list[TensorBlob] = []
            for a in c.state:
                if a is None:
                    continue
                with mx.stream(mx.Device(mx.cpu)):
                    a_cpu = mx.array(a)
                    mx.eval(a_cpu)
                blobs.append(
                    TensorBlob(
                        dtype=mx_dtype_to_str(a_cpu.dtype),
                        shape=tuple(int(d) for d in a_cpu.shape),
                        data=array_to_bytes(a_cpu),
                    )
                )
            if blobs:
                write_arrays_state(stream, layer_idx, blobs)
    return tokens_sent


def chunk_to_mlx_nhd(chunk: KVChunk) -> tuple[mx.array, mx.array]:
    shape = chunk.shape
    return (
        bytes_to_array(chunk.keys, shape, chunk.dtype),
        bytes_to_array(chunk.values, shape, chunk.dtype),
    )


def blob_to_mlx(blob: TensorBlob) -> mx.array:
    return bytes_to_array(blob.data, blob.shape, blob.dtype)


def inject_kv_chunk(
    cache: KVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
    *,
    start_pos: int = 0,
    existing_k: mx.array | None = None,
    existing_v: mx.array | None = None,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    if start_pos > 0 and existing_k is not None and existing_v is not None:
        cache.keys = mx.concatenate([existing_k[:, :, :start_pos, :], k_bhsd], axis=2)
        cache.values = mx.concatenate([existing_v[:, :, :start_pos, :], v_bhsd], axis=2)
    else:
        cache.keys = k_bhsd
        cache.values = v_bhsd
    cache.offset = offset


def inject_rotating_kv_chunk(
    cache: RotatingKVCache,
    keys_nhd: mx.array,
    values_nhd: mx.array,
    offset: int,
) -> None:
    k_bhsd = nhd_to_bhsd(keys_nhd)
    v_bhsd = nhd_to_bhsd(values_nhd)
    cache.keys = k_bhsd
    cache.values = v_bhsd
    cache.offset = offset
    cache._idx = int(k_bhsd.shape[2])


def inject_arrays_cache(cache: ArraysCache, blobs: list[TensorBlob]) -> None:
    cache.state = [blob_to_mlx(b) for b in blobs]


def serialize_mlx_cache_to_payload(
    caches: list[AnyMLXCache] | Sequence[AnyMLXCache],
    *,
    dtype: DType,
    model_id: str = "",
    request_id: str = "",
    start_pos: int = 0,
    max_tokens: int | None = None,
) -> bytes:
    buf = io.BytesIO()
    header = make_header(
        num_layers=len(caches),
        dtype=dtype,
        model_id=model_id,
        request_id=request_id,
        start_pos=start_pos,
    )
    write_header(buf, header)
    tokens_sent = send_mlx_kv_cache(buf, caches, max_tokens=max_tokens)
    write_done(buf, tokens_sent)
    return buf.getvalue()
