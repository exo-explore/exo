import threading
import time
from collections.abc import Callable
from typing import cast, final

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import Header, KVChunk
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import CacheSnapshot, snapshot_ssm_states
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger


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
    num_layers: int = 0

    def _on_header(header: Header) -> None:
        nonlocal num_layers
        num_layers = header.num_layers

    def _on_chunk(_chunk: KVChunk, chunks_received: int) -> None:
        nonlocal num_layers
        if on_prefill_progress is None:
            return
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
    return tps, num_tokens, [snapshot_ssm_states(cache)]


@final
class RemotePrefillHandle:
    """A remote prefill fetched on a background thread and ingested later.

    The thread only does socket I/O. MLX is not thread safe, so ingest() must run on
    the engine thread. Errors from the fetch are kept and raised by ingest(), where
    the caller falls back to local prefill.
    """

    def __init__(
        self,
        prompt_tokens: mx.array,
        *,
        endpoint: str,
        request_id: str,
        model_id: str,
        start_pos: int = 0,
    ) -> None:
        self._started = time.perf_counter()
        self._received = self._started
        self._start_pos = start_pos
        self._result: PrefillResult | None = None
        self._error: Exception | None = None
        request = PrefillRequest(
            model_id=model_id,
            token_ids=cast(list[int], prompt_tokens.tolist()),
            start_pos=start_pos,
            request_id=request_id,
        )
        self._thread = threading.Thread(
            target=self._fetch,
            args=(endpoint, request),
            name=f"remote-prefill-{request_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _fetch(self, endpoint: str, request: PrefillRequest) -> None:
        try:
            self._result = remote_prefill_fetch(endpoint, request)
        except Exception as error:
            self._error = error
        self._received = time.perf_counter()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def ingest(self, cache: KVCacheType) -> tuple[float, int, list[CacheSnapshot]]:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        caches = cast(list[KVCache | RotatingKVCache | ArraysCache], list(cache))
        final_offset = ingest_into_mlx_cache(
            self._result, caches, start_pos=self._start_pos
        )
        finished = time.perf_counter()
        num_tokens = final_offset - self._start_pos
        tps = num_tokens / max(finished - self._started, 0.001)
        logger.info(
            f"Remote prefill (background): {num_tokens} tokens at {tps:.0f} tok/s, "
            f"transfer={(self._received - self._started) * 1000:.0f}ms, "
            f"inject={(finished - self._received) * 1000:.0f}ms"
        )
        return tps, num_tokens, [snapshot_ssm_states(cache)]
