import json
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
    TensorBlob,
    read_header,
    read_message,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    chunk_to_mlx_nhd,
    inject_arrays_cache,
    inject_kv_chunk,
    inject_rotating_kv_chunk,
)

DEFAULT_PORT = 8900
_SOCKET_TIMEOUT_SECS = 60
_RECV_BUFFER_BYTES = 4 * 1024 * 1024


@dataclass
class PrefillRequest:
    model_id: str
    token_ids: list[int]
    start_pos: int = 0
    request_id: str = ""


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


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    return endpoint, DEFAULT_PORT


def remote_prefill_fetch(
    endpoint: str,
    request: PrefillRequest,
    on_header: Callable[[Header], None] | None = None,
    on_kv_chunk: Callable[[KVChunk, int], None] | None = None,
    timeout_secs: float = _SOCKET_TIMEOUT_SECS,
) -> PrefillResult:
    host, port = _parse_endpoint(endpoint)
    logger.info(
        f"Connecting to prefill server at {host}:{port} "
        f"({len(request.token_ids)} tokens, start_pos={request.start_pos})"
    )

    sock = socket.create_connection((host, port), timeout=timeout_secs)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER_BYTES)
    try:
        req_bytes = (
            json.dumps(
                {
                    "request_id": request.request_id,
                    "model": request.model_id,
                    "token_ids": request.token_ids,
                    "start_pos": request.start_pos,
                }
            ).encode("utf-8")
            + b"\n"
        )
        sock.sendall(req_bytes)

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
                break
            if isinstance(msg, KVChunk):
                kv_by_layer[msg.layer_idx].append(msg)
                chunks_received += 1
                if on_kv_chunk is not None:
                    on_kv_chunk(msg, chunks_received)
            elif isinstance(msg, ArraysState):
                result.arrays[msg.layer_idx] = msg.arrays
            elif isinstance(msg, Done):
                result.total_tokens = msg.total_tokens
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
) -> int:
    max_received = max(
        (sum(c.num_tokens for c in chunks) for chunks in result.kv_chunks.values()),
        default=0,
    )
    final_offset = start_pos + max_received

    for i, cache in enumerate(caches):
        if i in result.kv_chunks:
            chunks = result.kv_chunks[i]
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

        if i in result.arrays and isinstance(cache, ArraysCache):
            inject_arrays_cache(cache, result.arrays[i])

    return final_offset
