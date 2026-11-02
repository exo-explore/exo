import math
import time
from collections.abc import Callable
from typing import cast, final

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import Header, KVChunk
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import CacheSnapshot, snapshot_ssm_states
from exo.worker.engines.mlx.disaggregated.client import (
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger

# While a remote prefill runs, its progress reports are the runner's only sign of life: the
# runner supervisor stops a generating runner that reports nothing for
# RUNNER_STALL_TIMEOUT (90 s), and a remote prefill node can stream its KV for longer than
# that (a long prompt, a cold prefill node). So progress is reported when the stream's
# header arrives and as its chunks arrive, at most once per this many seconds. Every wait in
# between is bounded by the client's per-receive socket timeout (60 s), after which the
# fetch fails and the caller prefills locally: a slow remote prefill keeps reporting or
# falls back, and is never mistaken for a stuck runner.
PROGRESS_INTERVAL_SECONDS = 1.0


@final
class RemotePrefillProgress:
    """How much of a remote prefill's KV stream has arrived.

    The stream holds every layer's KV for the prompt, in one or more chunks per layer, so
    the tokens received over all layers divided by the number of layers estimates how
    much of the prompt has arrived, whatever the chunking. Layers sent as ArraysState
    carry no tokens, so for models with them the estimate stays below the total.
    """

    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens
        self._num_layers = 0
        self._layer_tokens = 0

    def on_header(self, header: Header) -> None:
        self._num_layers = header.num_layers

    def on_kv_chunk(self, chunk: KVChunk, _chunks_received: int) -> None:
        self._layer_tokens += chunk.num_tokens

    @property
    def processed_tokens(self) -> int:
        if self._num_layers <= 0:
            return 0
        return min(self._layer_tokens // self._num_layers, self.total_tokens)


def remote_prefill(
    prompt_tokens: mx.array,
    cache: KVCacheType,
    on_prefill_progress: Callable[[int, int], None] | None,
    *,
    endpoint: str,
    request_id: str,
    model_id: str,
    start_pos: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    t0 = time.perf_counter()
    total_prompt_tokens = int(prompt_tokens.shape[0])
    progress = RemotePrefillProgress(total_prompt_tokens)
    last_report = -math.inf

    def _report(force: bool) -> None:
        nonlocal last_report
        if on_prefill_progress is None:
            return
        now = time.monotonic()
        if force or now - last_report >= PROGRESS_INTERVAL_SECONDS:
            last_report = now
            on_prefill_progress(progress.processed_tokens, total_prompt_tokens)

    def _on_header(header: Header) -> None:
        progress.on_header(header)
        _report(force=True)

    def _on_chunk(chunk: KVChunk, chunks_received: int) -> None:
        progress.on_kv_chunk(chunk, chunks_received)
        _report(force=False)

    request = PrefillRequest(
        model_id=model_id,
        token_ids=cast(list[int], prompt_tokens.tolist()),
        start_pos=start_pos,
        request_id=request_id,
    )
    _report(force=True)
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
    return tps, num_tokens, [snapshot_ssm_states(cache)]
