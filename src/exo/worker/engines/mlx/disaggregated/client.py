import socket
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import BinaryIO, cast

import mlx.core as mx
from loguru import logger
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    KVChunk,
    ProtocolError,
    TensorBlob,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest, write_request
from exo.worker.engines.mlx.disaggregated.adapter import (
    blob_to_mlx,
    chunk_to_mlx_nhd,
    inject_arrays_cache,
    inject_kv_chunk,
    inject_rotating_kv_chunk,
)

_SOCKET_TIMEOUT_SECS = 60
_RECV_BUFFER_BYTES = 4 * 1024 * 1024


@dataclass
class PrefillResult:
    header: Header
    kv_chunks: dict[int, list[KVChunk]] = field(
        default_factory=dict[int, list[KVChunk]]
    )
    arrays: dict[int, list[TensorBlob]] = field(
        default_factory=dict[int, list[TensorBlob]]
    )
    total_tokens: int = 0


@dataclass
class _StagedKV:
    keys: list[mx.array] = field(default_factory=list[mx.array])
    values: list[mx.array] = field(default_factory=list[mx.array])
    num_tokens: int = 0


IngestCache = KVCache | RotatingKVCache | ArraysCache


class LayerwiseIngest:
    """Per-layer ingest of a remote prefill stream.

    Each KVChunk (and the ArraysState of an ArraysCache layer) is decoded into MLX arrays
    as soon as it arrives, so its wire bytes are dropped at once and the decoding overlaps
    the rest of the transfer instead of following it. Nothing is written into the caller's
    caches until commit(), which requires the stream's Done: a stream that fails part way
    (an error frame, a closed socket, a truncated frame) leaves the caches exactly as they
    were, and the caller prefills locally from them.

    The prefill server numbers layers globally. A pipeline rank holds the layers
    ``[layer_offset, layer_offset + num_layers)``; layers outside that range are dropped
    on arrival without being decoded, and local cache i takes global layer
    i + layer_offset at commit.

    commit() applies the staged layers with the same operations as
    ingest_into_mlx_cache(), so the resulting cache is identical to buffered ingest.
    This makes MLX calls: use it on the thread that owns the model's MLX state, never on
    the background fetch thread.
    """

    def __init__(
        self,
        *,
        layer_offset: int = 0,
        num_layers: int | None = None,
    ) -> None:
        self._layer_offset = layer_offset
        self._num_layers = num_layers
        self._kv: dict[int, _StagedKV] = {}
        self._arrays: dict[int, list[mx.array]] = {}
        # tokens per layer for every layer in the stream, kept or not: the offset is the
        # longest layer's, as in buffered ingest
        self._tokens: dict[int, int] = defaultdict(int)
        self.total_tokens: int | None = None

    @classmethod
    def for_caches(
        cls, caches: list[IngestCache], *, layer_offset: int = 0
    ) -> "LayerwiseIngest":
        """Staging for this rank's caches (read only to learn their kinds and count)."""
        return cls(layer_offset=layer_offset, num_layers=len(caches))

    @property
    def done(self) -> bool:
        return self.total_tokens is not None

    def _kept(self, layer_idx: int) -> bool:
        if layer_idx < self._layer_offset:
            return False
        return (
            self._num_layers is None
            or layer_idx < self._layer_offset + self._num_layers
        )

    def add_kv_chunk(self, chunk: KVChunk) -> None:
        if self.done:
            raise ProtocolError(f"KV chunk for layer {chunk.layer_idx} after Done")
        self._tokens[chunk.layer_idx] += chunk.num_tokens
        if not self._kept(chunk.layer_idx):
            return
        keys, values = chunk_to_mlx_nhd(chunk)
        mx.eval(keys, values)
        staged = self._kv.setdefault(chunk.layer_idx, _StagedKV())
        staged.keys.append(keys)
        staged.values.append(values)
        staged.num_tokens += chunk.num_tokens

    def add_arrays_state(self, state: ArraysState) -> None:
        if self.done:
            raise ProtocolError(f"arrays state for layer {state.layer_idx} after Done")
        if not self._kept(state.layer_idx):
            return
        arrays = [blob_to_mlx(b) for b in state.arrays]
        mx.eval(*arrays)
        self._arrays[state.layer_idx] = arrays

    def finish(self, done: Done) -> None:
        self.total_tokens = done.total_tokens

    def commit(self, caches: list[IngestCache], *, start_pos: int = 0) -> int:
        """Write the staged layers into ``caches``; all of them or, on error, none."""
        if not self.done:
            raise ProtocolError("Remote prefill stream ended before Done")
        final_offset = start_pos + max(self._tokens.values(), default=0)
        undo: list[Callable[[], None]] = []
        try:
            for local_index, cache in enumerate(caches):
                layer_index = local_index + self._layer_offset
                staged = self._kv.get(layer_index)
                if staged is not None:
                    if len(staged.keys) == 1:
                        k_nhd, v_nhd = staged.keys[0], staged.values[0]
                    else:
                        k_nhd = mx.concatenate(staged.keys, axis=0)
                        v_nhd = mx.concatenate(staged.values, axis=0)
                    if isinstance(cache, RotatingKVCache):
                        undo.append(_saved_rotating(cache))
                        inject_rotating_kv_chunk(cache, k_nhd, v_nhd, final_offset)
                    elif isinstance(cache, KVCache):
                        undo.append(_saved_kv(cache))
                        if start_pos > 0:
                            inject_kv_chunk(
                                cache,
                                k_nhd,
                                v_nhd,
                                final_offset,
                                start_pos=start_pos,
                                existing_k=cache.keys,
                                existing_v=cache.values,
                            )
                        else:
                            inject_kv_chunk(cache, k_nhd, v_nhd, final_offset)

                arrays = self._arrays.get(layer_index)
                if arrays is not None and isinstance(cache, ArraysCache):
                    undo.append(_saved_arrays(cache))
                    # what inject_arrays_cache() does, with the arrays decoded already
                    cache.cache = list(arrays)
        except Exception:
            for restore in reversed(undo):
                restore()
            raise
        return final_offset


def _saved_kv(cache: KVCache) -> Callable[[], None]:
    keys, values, offset = cache.keys, cache.values, cache.offset

    def restore() -> None:
        cache.keys, cache.values, cache.offset = keys, values, offset

    return restore


def _saved_rotating(cache: RotatingKVCache) -> Callable[[], None]:
    keys, values, offset, idx = cache.keys, cache.values, cache.offset, cache._idx

    def restore() -> None:
        cache.keys, cache.values, cache.offset = keys, values, offset
        cache._idx = idx

    return restore


def _saved_arrays(cache: ArraysCache) -> Callable[[], None]:
    arrays = cache.cache

    def restore() -> None:
        cache.cache = arrays

    return restore


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    raise ValueError(f"Invalid endpoint {endpoint}")


def remote_prefill_fetch(
    endpoint: str,
    request: PrefillRequest,
    on_header: Callable[[Header], None] | None = None,
    on_kv_chunk: Callable[[KVChunk, int], None] | None = None,
    timeout_secs: float = _SOCKET_TIMEOUT_SECS,
    layerwise: LayerwiseIngest | None = None,
) -> PrefillResult:
    """Fetch a remote prefill. With ``layerwise``, layers are handed to it as they arrive
    (decoded with MLX on this thread) instead of being buffered in the result, and a
    stream that ends without Done raises."""
    host, port = _parse_endpoint(endpoint)
    logger.info(
        f"Connecting to prefill server at {host}:{port} "
        f"({len(request.token_ids)} tokens, start_pos={request.start_pos})"
    )

    sock = socket.create_connection((host, port), timeout=timeout_secs)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER_BYTES)
    try:
        wfile = sock.makefile("wb", buffering=256 * 1024)
        wstream: BinaryIO = cast(BinaryIO, cast(object, wfile))
        write_request(wstream, request)

        raw_stream = sock.makefile("rb", buffering=256 * 1024)
        stream: BinaryIO = cast(BinaryIO, cast(object, raw_stream))

        header = read_header(stream)
        if on_header is not None:
            on_header(header)

        result = PrefillResult(header=header)
        kv_by_layer: dict[int, list[KVChunk]] = defaultdict(list)
        chunks_received = 0

        while True:
            msg = read_message(stream)
            if msg is None:
                if layerwise is not None:
                    raise ConnectionError("Remote prefill stream ended before Done")
                break
            if isinstance(msg, KVChunk):
                if layerwise is not None:
                    layerwise.add_kv_chunk(msg)
                else:
                    kv_by_layer[msg.layer_idx].append(msg)
                chunks_received += 1
                if on_kv_chunk is not None:
                    on_kv_chunk(msg, chunks_received)
            elif isinstance(msg, ArraysState):
                if layerwise is not None:
                    layerwise.add_arrays_state(msg)
                else:
                    result.arrays[msg.layer_idx] = msg.arrays
            elif isinstance(msg, Done):
                result.total_tokens = msg.total_tokens
                if layerwise is not None:
                    layerwise.finish(msg)
                break
            else:
                raise RuntimeError(f"Prefill server error [{msg.code}]: {msg.message}")

        result.kv_chunks = dict(kv_by_layer)
        return result
    finally:
        sock.close()


def ingest_into_mlx_cache(
    result: PrefillResult,
    caches: list[KVCache | RotatingKVCache | ArraysCache],
    *,
    start_pos: int = 0,
    layer_offset: int = 0,
) -> int:
    """Inject a remote prefill into this rank's caches and return the new offset.

    The prefill server numbers layers globally. A pipeline rank holds only its own
    layers, so local cache i is global layer i + layer_offset (the shard's
    start_layer).
    """
    max_received = max(
        (sum(c.num_tokens for c in chunks) for chunks in result.kv_chunks.values()),
        default=0,
    )
    final_offset = start_pos + max_received

    for local_index, cache in enumerate(caches):
        layer_index = local_index + layer_offset
        if layer_index in result.kv_chunks:
            chunks = result.kv_chunks[layer_index]
            if len(chunks) == 1:
                k_nhd, v_nhd = chunk_to_mlx_nhd(chunks[0])
            else:
                decoded = [chunk_to_mlx_nhd(c) for c in chunks]
                k_nhd = mx.concatenate([k for k, _ in decoded], axis=0)
                v_nhd = mx.concatenate([v for _, v in decoded], axis=0)

            if isinstance(cache, RotatingKVCache):
                inject_rotating_kv_chunk(cache, k_nhd, v_nhd, final_offset)
            elif isinstance(cache, KVCache):
                if start_pos > 0:
                    inject_kv_chunk(
                        cache,
                        k_nhd,
                        v_nhd,
                        final_offset,
                        start_pos=start_pos,
                        existing_k=cache.keys,
                        existing_v=cache.values,
                    )
                else:
                    inject_kv_chunk(cache, k_nhd, v_nhd, final_offset)

        if layer_index in result.arrays and isinstance(cache, ArraysCache):
            inject_arrays_cache(cache, result.arrays[layer_index])

    return final_offset
