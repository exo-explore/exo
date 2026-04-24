import time
from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.disaggregated.client import (
    PrefillRequest,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.disaggregated.protocol import KVChunk
from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.cache import CacheSnapshot
from exo.worker.runner.bootstrap import logger


def remote_prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    *,
    endpoint: str,
    request_id: str,
    model_id: str,
    start_pos: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    del model, tokenizer, sampler, group, distributed_prompt_progress_callback

    t0 = time.perf_counter()
    total_prompt_tokens = int(prompt_tokens.shape[0])
    num_layers_box: list[int] = [0]

    def _on_header(header: dict[str, object]) -> None:
        nl = header.get("num_layers", 0)
        if isinstance(nl, int):
            num_layers_box[0] = nl

    def _on_chunk(_chunk: KVChunk, chunks_received: int) -> None:
        if on_prefill_progress is None:
            return
        num_layers = num_layers_box[0]
        if num_layers > 0 and chunks_received % num_layers == 0:
            tokens_so_far = chunks_received // num_layers
            on_prefill_progress(
                min(tokens_so_far, total_prompt_tokens),
                total_prompt_tokens,
            )

    request = PrefillRequest(
        model_id=model_id,
        token_ids=cast(list[int], prompt_tokens.tolist()),
        start_pos=start_pos,
        request_id=request_id,
    )
    result = remote_prefill_fetch(
        endpoint, request, on_header=_on_header, on_kv_chunk=_on_chunk
    )
    t_received = time.perf_counter()

    caches = cast(list[KVCache | RotatingKVCache | ArraysCache], list(cache))
    final_offset = ingest_into_mlx_cache(result, caches, start_pos=start_pos)
    t_done = time.perf_counter()

    num_tokens = final_offset - start_pos
    tps = num_tokens / max(t_done - t0, 0.001)

    logger.info(
        f"Remote prefill: {num_tokens} tokens (start_pos={start_pos}, "
        f"final_offset={final_offset}) at {tps:.0f} tok/s, "
        f"transfer={(t_received - t0) * 1000:.0f}ms, "
        f"inject={(t_done - t_received) * 1000:.0f}ms"
    )
    return tps, num_tokens, []
