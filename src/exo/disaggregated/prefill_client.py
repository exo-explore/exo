from __future__ import annotations

import json
import socket
import time
from collections import defaultdict
from typing import TYPE_CHECKING, BinaryIO, cast

import mlx.core as mx
import torch
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.disaggregated.protocol import (
    ArraysState,
    Done,
    KVChunk,
    read_header,
    read_message,
)

if TYPE_CHECKING:
    from exo.shared.types.mlx import Model

from exo.worker.runner.bootstrap import logger


def _torch_to_mx(t: torch.Tensor) -> mx.array:
    t_cpu: torch.Tensor = t.detach().cpu()
    if t_cpu.dtype == torch.bfloat16:
        return mx.array(t_cpu.float().numpy()).astype(mx.bfloat16)  # pyright: ignore[reportAny]
    return mx.array(t_cpu.numpy())  # pyright: ignore[reportAny]


def _nhd_to_bhsd(keys: torch.Tensor, values: torch.Tensor) -> tuple[mx.array, mx.array]:
    k_mx = _torch_to_mx(keys.permute(1, 0, 2).unsqueeze(0))
    v_mx = _torch_to_mx(values.permute(1, 0, 2).unsqueeze(0))
    return k_mx, v_mx


def _inject_kv_cache(cache: KVCache, keys: torch.Tensor, values: torch.Tensor, num_tokens: int) -> None:
    k_mx, v_mx = _nhd_to_bhsd(keys, values)
    cache.keys = k_mx
    cache.values = v_mx
    cache.offset = num_tokens


def _inject_rotating_kv_cache(cache: RotatingKVCache, keys: torch.Tensor, values: torch.Tensor, num_tokens: int) -> None:
    k_mx, v_mx = _nhd_to_bhsd(keys, values)
    seq_len = int(k_mx.shape[2])

    if seq_len <= cache.max_size:
        cache.keys = k_mx
        cache.values = v_mx
        cache.offset = seq_len
        cache._idx = seq_len
    else:
        keep = cache.keep
        window = cache.max_size
        sink_keys = k_mx[:, :, :keep, :]
        sink_values = v_mx[:, :, :keep, :]
        recent_keys = k_mx[:, :, -(window - keep):, :]
        recent_values = v_mx[:, :, -(window - keep):, :]
        cache.keys = mx.concatenate([sink_keys, recent_keys], axis=2)
        cache.values = mx.concatenate([sink_values, recent_values], axis=2)
        cache.offset = num_tokens
        cache._idx = keep


def _inject_arrays_cache(cache: ArraysCache, arrays: list[torch.Tensor]) -> None:
    cache.state = [_torch_to_mx(arr) for arr in arrays]


def remote_prefill(
    endpoint: str,
    token_ids: list[int],
    model_id: str,
    mlx_model: Model,
) -> tuple[list[KVCache | RotatingKVCache | ArraysCache], int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
    else:
        host = endpoint
        port = 8900

    logger.info(f"Connecting to prefill server at {host}:{port} ({len(token_ids)} tokens)")
    t0 = time.perf_counter()

    sock = socket.create_connection((host, port), timeout=30)
    try:
        request = json.dumps({"model": model_id, "token_ids": token_ids}).encode("utf-8") + b"\n"
        sock.sendall(request)

        raw_stream = sock.makefile("rb", buffering=65536)
        stream: BinaryIO = raw_stream  # pyright: ignore[reportAssignmentType]

        first_byte: bytes = raw_stream.peek(1)[:1]  # type: ignore
        if first_byte == b"{":
            line = stream.readline()
            error_resp: dict[str, object] = json.loads(line.decode("utf-8"))  # pyright: ignore[reportAny]
            raise RuntimeError(f"Prefill server error: {error_resp.get('error', 'unknown')}")

        header = read_header(stream)

        kv_buffers: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        arrays_buffers: dict[int, list[torch.Tensor]] = {}
        total_tokens = 0

        t_first_chunk = None
        while True:
            msg = read_message(stream, header)
            if msg is None:
                break

            if isinstance(msg, KVChunk):
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                kv_buffers[msg.layer_idx].append((msg.keys, msg.values))
            elif isinstance(msg, ArraysState):
                arrays_buffers[msg.layer_idx] = msg.arrays
            elif isinstance(msg, Done):  # pyright: ignore[reportUnnecessaryIsInstance]
                total_tokens = msg.total_tokens
                break

        t_received = time.perf_counter()
    finally:
        sock.close()

    caches: list[KVCache | RotatingKVCache | ArraysCache] = cast(list[KVCache | RotatingKVCache | ArraysCache], mlx_model.make_cache())  # pyright: ignore[reportUnknownMemberType]

    for i, cache in enumerate(caches):
        if i in kv_buffers:
            chunks = kv_buffers[i]
            all_keys: torch.Tensor
            all_values: torch.Tensor
            if len(chunks) == 1:
                all_keys, all_values = chunks[0]
            else:
                all_keys = torch.cat([k for k, _v in chunks], dim=0)  # type: ignore
                all_values = torch.cat([v for _k, v in chunks], dim=0)  # type: ignore

            if isinstance(cache, RotatingKVCache):
                _inject_rotating_kv_cache(cache, all_keys, all_values, total_tokens)  # pyright: ignore[reportUnknownArgumentType]
            elif isinstance(cache, KVCache):
                _inject_kv_cache(cache, all_keys, all_values, total_tokens)  # pyright: ignore[reportUnknownArgumentType]

        if i in arrays_buffers and isinstance(cache, ArraysCache):
            _inject_arrays_cache(cache, arrays_buffers[i])

    t_injected = time.perf_counter()
    logger.info(
        f"Remote prefill: {total_tokens} tokens, "
        f"transfer={((t_received - t0) * 1000):.0f}ms, "
        f"inject={((t_injected - t_received) * 1000):.0f}ms, "
        f"total={((t_injected - t0) * 1000):.0f}ms"
    )

    return caches, total_tokens
