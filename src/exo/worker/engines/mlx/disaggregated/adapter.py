from typing import BinaryIO, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from exo.worker.disaggregated.protocol import (
    DType,
    Header,
    KVChunk,
    TensorBlob,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)
from exo.worker.engines.mlx.models.deepseek_v4.model import DeepseekV4Cache
from exo.worker.engines.mlx.models.deepseek_v4.model import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from exo.worker.engines.mlx.types import KVCacheType, SnapshotCache
from exo.worker.runner.bootstrap import logger

_STR_TO_MX: dict[DType, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}

_MX_TO_STR: dict[mx.Dtype, DType] = {v: k for k, v in _STR_TO_MX.items()}


def mx_dtype_to_str(dtype: mx.Dtype) -> DType:
    if dtype not in _MX_TO_STR:
        raise ValueError(f"Unsupported mlx dtype on wire: {dtype}")
    return _MX_TO_STR[dtype]


def wire_dtype_from_cache(caches: KVCacheType) -> DType:
    for c in caches:
        keys: mx.array | None = getattr(c, "keys", None)
        if keys is None:
            continue
        if keys.dtype in _MX_TO_STR:
            return _MX_TO_STR[keys.dtype]
        break
    return "bfloat16"


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
    match dtype:
        case "bfloat16":
            arr = np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()
            return mx.array(arr).view(mx.bfloat16)
        case "float16":
            arr = np.frombuffer(data, dtype=np.float16).reshape(shape).copy()
            return mx.array(arr)
        case "float32":
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
            return mx.array(arr)


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
    caches: KVCacheType,
    *,
    dtype: DType,
    start_pos: int = 0,
    max_tokens: int | None = None,
) -> int:
    tokens_sent = 0
    for layer_idx, c in enumerate(caches):
        match c:
            case DeepseekV4Cache():
                write_arrays_state(stream, layer_idx, deepseek_v4_cache_to_blobs(c))
                tokens_sent = max(tokens_sent, c.offset - start_pos)
            case QuantizedKVCache() | CacheList() | SnapshotCache():
                raise NotImplementedError
            case KVCache() | RotatingKVCache():
                keys = c.keys
                values = c.values
                if keys is None or values is None:
                    continue
                offset = int(c.offset)
                if max_tokens is not None:
                    offset = min(offset, max_tokens)
                if offset <= start_pos:
                    continue
                with mx.stream(mx.Device(mx.cpu)):
                    k = mx.array(keys[:, :, start_pos:offset, :])
                    v = mx.array(values[:, :, start_pos:offset, :])
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
                    dtype=dtype,
                    keys=array_to_bytes(k_nhd),
                    values=array_to_bytes(v_nhd),
                )
                if tokens_sent != 0 and num_tokens != tokens_sent:
                    logger.critical(
                        f"Unexpected number of tokens sent {num_tokens} != {tokens_sent}"
                    )
                tokens_sent = num_tokens
            case ArraysCache():
                blobs: list[TensorBlob] = []
                for a in c.cache:
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


_DEEPSEEK_V4_WIRE_VERSION = 1
# Same keys as DeepseekV4Cache._branches; the order is the order on the wire.
_DEEPSEEK_V4_BRANCH_KEYS = ("compressor", "indexer")
_DEEPSEEK_V4_ARRAYS_PER_BRANCH = 5
_DEEPSEEK_V4_HEADER_FIELDS_PER_BRANCH = _DEEPSEEK_V4_ARRAYS_PER_BRANCH + 3


def _compressor_branch_arrays(branch: CompressorBranch) -> list[mx.array | None]:
    return [
        branch.buffer_kv,
        branch.buffer_gate,
        branch.prev_kv,
        branch.prev_gate,
        branch.pool,
    ]


def _window_arrays(window: RotatingKVCache) -> tuple[mx.array, mx.array]:
    """The window's keys and values, cut to its offset while it is still filling."""
    keys, values = window.keys, window.values
    assert keys is not None and values is not None
    if window.offset < keys.shape[2]:
        return keys[..., : window.offset, :], values[..., : window.offset, :]
    return keys, values


def deepseek_v4_cache_to_blobs(cache: DeepseekV4Cache) -> list[TensorBlob]:
    """Serialise one DeepseekV4Cache (batch size 1) as the blobs of one ArraysState.

    A DeepseekV4Cache is a sliding window (a RotatingKVCache) plus two compressor
    branches, "compressor" and "indexer". The whole state is sent whatever start_pos
    is, because the cache is not trimmable. The layout reuses the existing wire
    format unchanged:

    blob 0, float32 header:
        version, keep, max_size, offset, _idx, window_present,
        then for each branch: five presence flags (buffer_kv, buffer_gate, prev_kv,
        prev_gate, pool), buffer_count, len(buffer_lengths) or -1,
        len(pool_lengths) or -1,
        then for each branch: its buffer_lengths, then its pool_lengths.
    remaining blobs: window keys and values if present, then each present branch
    array in the order above.

    Every header value is an integer; float32 holds them exactly below 2**24.
    """
    window = cache.local
    window_present = not cache.empty()
    header: list[int] = [
        _DEEPSEEK_V4_WIRE_VERSION,
        window.keep,
        window.max_size,
        window.offset,
        window._idx,
        1 if window_present else 0,
    ]
    arrays: list[mx.array] = []
    if window_present:
        arrays += list(_window_arrays(window))

    lengths: list[int] = []
    for key in _DEEPSEEK_V4_BRANCH_KEYS:
        branch = cache._branches[key]
        for array in _compressor_branch_arrays(branch):
            header.append(0 if array is None else 1)
            if array is not None:
                arrays.append(array)
        header.append(branch.buffer_count or 0)
        for branch_lengths in (branch.buffer_lengths, branch.pool_lengths):
            if branch_lengths is None:
                header.append(-1)
            else:
                header.append(len(branch_lengths))
                lengths += branch_lengths

    blobs = [
        TensorBlob(
            dtype="float32",
            shape=(len(header) + len(lengths),),
            data=np.asarray(header + lengths, dtype=np.float32).tobytes(),
        )
    ]
    with mx.stream(mx.Device(mx.cpu)):
        for array in arrays:
            # The wire carries bf16, fp16 and fp32 only.
            if array.dtype not in _MX_TO_STR:
                array = array.astype(mx.float32)
            array_cpu = mx.array(array)
            mx.eval(array_cpu)
            blobs.append(
                TensorBlob(
                    dtype=mx_dtype_to_str(array_cpu.dtype),
                    shape=tuple(int(d) for d in array_cpu.shape),
                    data=array_to_bytes(array_cpu),
                )
            )
    return blobs


def _read_lengths(
    header: list[int], position: int, count: int
) -> tuple[list[int] | None, int]:
    if count < 0:
        return None, position
    return header[position : position + count], position + count


def inject_deepseek_v4_cache(cache: DeepseekV4Cache, blobs: list[TensorBlob]) -> None:
    """Rebuild a DeepseekV4Cache from the blobs of deepseek_v4_cache_to_blobs.

    Raises ValueError for an unknown wire version; the caller treats it like any
    other failed remote prefill.
    """
    header_values = np.frombuffer(blobs[0].data, dtype=np.float32)
    header = cast(list[int], header_values.astype(np.int64).tolist())
    if header[0] != _DEEPSEEK_V4_WIRE_VERSION:
        raise ValueError(f"Unknown DeepseekV4Cache wire version {header[0]}")
    keep, max_size, offset, window_index, window_present = header[1:6]
    arrays = iter([blob_to_mlx(blob) for blob in blobs[1:]])

    window = cache.local
    if window_present:
        window.keys = next(arrays)
        window.values = next(arrays)
    else:
        window.keys = None
        window.values = None
    window.keep = keep
    window.max_size = max_size
    window.offset = offset
    window._idx = window_index

    position = 6
    branches: list[tuple[CompressorBranch, int, int]] = []
    for _ in _DEEPSEEK_V4_BRANCH_KEYS:
        fields = header[position : position + _DEEPSEEK_V4_HEADER_FIELDS_PER_BRANCH]
        position += _DEEPSEEK_V4_HEADER_FIELDS_PER_BRANCH
        present = fields[:_DEEPSEEK_V4_ARRAYS_PER_BRANCH]
        buffer_count, buffer_lengths_count, pool_lengths_count = fields[
            _DEEPSEEK_V4_ARRAYS_PER_BRANCH:
        ]
        branch = CompressorBranch()
        (
            branch.buffer_kv,
            branch.buffer_gate,
            branch.prev_kv,
            branch.prev_gate,
            branch.pool,
        ) = (next(arrays) if flag else None for flag in present)
        branch.buffer_count = buffer_count
        branches.append((branch, buffer_lengths_count, pool_lengths_count))

    for branch, buffer_lengths_count, pool_lengths_count in branches:
        branch.buffer_lengths, position = _read_lengths(
            header, position, buffer_lengths_count
        )
        branch.pool_lengths, position = _read_lengths(
            header, position, pool_lengths_count
        )

    cache._branches = {
        key: branch
        for key, (branch, _, _) in zip(_DEEPSEEK_V4_BRANCH_KEYS, branches, strict=True)
    }


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
    cache.cache = [blob_to_mlx(b) for b in blobs]


def write_cache_to_wire(
    wfile: BinaryIO,
    cache: KVCacheType,
    *,
    request_id: str = "",
    model_id: str = "",
    start_pos: int = 0,
) -> int:
    dtype = wire_dtype_from_cache(cache)
    write_header(
        wfile,
        Header(
            request_id=request_id,
            model_id=model_id,
            num_layers=len(cache),
            dtype=dtype,
            start_pos=start_pos,
        ),
    )
    tokens_sent = send_mlx_kv_cache(wfile, cache, dtype=dtype, start_pos=start_pos)
    write_done(wfile, tokens_sent)
    wfile.flush()
    return tokens_sent
