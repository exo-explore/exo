from __future__ import annotations

import json
import socket
import time
from collections import defaultdict
from collections.abc import Callable
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
    cache.keys = k_mx
    cache.values = v_mx
    cache.offset = num_tokens
    cache._idx = seq_len


def _inject_arrays_cache(cache: ArraysCache, arrays: list[torch.Tensor]) -> None:
    cache.state = [_torch_to_mx(arr) for arr in arrays]


def remote_prefill(
    endpoint: str,
    token_ids: list[int],
    model_id: str,
    mlx_model: Model,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    existing_cache: list[KVCache | RotatingKVCache | ArraysCache] | None = None,
    start_pos: int = 0,
) -> tuple[list[KVCache | RotatingKVCache | ArraysCache], int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
    else:
        host = endpoint
        port = 8900

    logger.info(f"Connecting to prefill server at {host}:{port} ({len(token_ids)} tokens, start_pos={start_pos})")
    t0 = time.perf_counter()

    sock = socket.create_connection((host, port), timeout=30)
    try:
        request = json.dumps({"model": model_id, "token_ids": token_ids, "start_pos": start_pos}).encode("utf-8") + b"\n"
        sock.sendall(request)

        raw_stream = sock.makefile("rb", buffering=65536)
        stream: BinaryIO = raw_stream  # pyright: ignore[reportAssignmentType]

        first_byte: bytes = raw_stream.peek(1)[:1]  # type: ignore
        if first_byte == b"{":
            line = stream.readline()
            error_resp: dict[str, object] = json.loads(line.decode("utf-8"))  # pyright: ignore[reportAny]
            raise RuntimeError(f"Prefill server error: {error_resp.get('error', 'unknown')}")

        header = read_header(stream)
        num_layers: int = header["num_layers"]  # pyright: ignore[reportAssignmentType]
        total_prompt_tokens = len(token_ids)

        kv_buffers: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        arrays_buffers: dict[int, list[torch.Tensor]] = {}
        total_tokens = 0
        layers_seen: set[int] = set()

        tokens_received = 0
        chunks_received = 0
        t_first_chunk = None
        while True:
            msg = read_message(stream, header)
            if msg is None:
                break

            if isinstance(msg, KVChunk):
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                kv_buffers[msg.layer_idx].append((msg.keys, msg.values))
                chunks_received += 1
                layers_seen.add(msg.layer_idx)
                tokens_received += msg.num_tokens
                if on_prefill_progress and num_layers > 0 and chunks_received % num_layers == 0:
                    on_prefill_progress(
                        min(tokens_received // num_layers, total_prompt_tokens - start_pos),
                        total_prompt_tokens - start_pos,
                    )
            elif isinstance(msg, ArraysState):
                arrays_buffers[msg.layer_idx] = msg.arrays
            elif isinstance(msg, Done):  # pyright: ignore[reportUnnecessaryIsInstance]
                total_tokens = msg.total_tokens
                break

        t_received = time.perf_counter()
    finally:
        sock.close()

    if existing_cache is not None and start_pos > 0:
        caches = existing_cache
    else:
        if hasattr(mlx_model, "make_cache"):
            caches = cast(list[KVCache | RotatingKVCache | ArraysCache], mlx_model.make_cache())  # pyright: ignore[reportUnknownMemberType]
        else:
            from mlx_lm.models.cache import make_prompt_cache
            caches = cast(list[KVCache | RotatingKVCache | ArraysCache], make_prompt_cache(mlx_model))  # pyright: ignore[reportUnknownMemberType]

    max_received = max((sum(k.shape[0] for k, _v in chunks) for chunks in kv_buffers.values()), default=0)
    final_offset = start_pos + max_received

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
                _inject_rotating_kv_cache(cache, all_keys, all_values, final_offset)  # pyright: ignore[reportUnknownArgumentType]
            elif isinstance(cache, KVCache):
                if start_pos > 0 and cache.keys is not None:
                    k_new, v_new = _nhd_to_bhsd(all_keys, all_values)  # pyright: ignore[reportUnknownArgumentType]
                    cache.keys = mx.concatenate([cache.keys[:, :, :start_pos, :], k_new], axis=2)
                    cache.values = mx.concatenate([cache.values[:, :, :start_pos, :], v_new], axis=2)
                    cache.offset = final_offset
                else:
                    _inject_kv_cache(cache, all_keys, all_values, final_offset)  # pyright: ignore[reportUnknownArgumentType]

        if i in arrays_buffers and isinstance(cache, ArraysCache):
            _inject_arrays_cache(cache, arrays_buffers[i])

    t_injected = time.perf_counter()
    logger.info(
        f"Remote prefill: {total_tokens} new tokens (start_pos={start_pos}, final_offset={final_offset}), "
        f"transfer={((t_received - t0) * 1000):.0f}ms, "
        f"inject={((t_injected - t_received) * 1000):.0f}ms, "
        f"total={((t_injected - t0) * 1000):.0f}ms"
    )

    return caches, final_offset
