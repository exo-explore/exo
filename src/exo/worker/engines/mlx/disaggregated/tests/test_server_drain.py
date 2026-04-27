"""End-to-end producer test: server thread receives request, main thread
drains queue, runs the prefill callable, returns bytes."""

import io
import queue
import threading
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.utils.ports import random_ephemeral_port
from exo.worker.engines.mlx.disaggregated.adapter import (
    chunk_to_mlx_nhd,
    serialize_mlx_cache_to_payload,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillRequest,
    PrefillResult,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.disaggregated.protocol import (
    Done,
    KVChunk,
    make_header,
    read_header,
    read_message,
    write_done,
    write_header,
)
from exo.worker.engines.mlx.disaggregated.server import (
    PrefillJob,
    PrefillServer,
)


def _equal(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _make_cache(seq_len: int, n_heads: int, head_dim: int) -> KVCache:
    mx.random.seed(0)
    cache = KVCache()
    cache.keys = (mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10).astype(
        mx.bfloat16
    )
    cache.values = (
        mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
    ).astype(mx.bfloat16)
    cache.offset = seq_len
    return cache


@pytest.mark.slow
def test_server_drains_via_main_thread() -> None:
    seq_len = 4
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    # Mimic the runner's queue-and-drain pattern.
    request_queue: queue.Queue[
        tuple[PrefillJob, threading.Event, list[bytes | None]]
    ] = queue.Queue()

    def resolve(job: PrefillJob) -> bytes | None:
        event = threading.Event()
        holder: list[bytes | None] = [None]
        request_queue.put((job, event, holder))
        if not event.wait(timeout=5):
            return None
        return holder[0]

    server = PrefillServer(
        resolve=resolve, host="127.0.0.1", port=random_ephemeral_port()
    )
    port = server.start()

    def serve_one() -> bytes:
        return serialize_mlx_cache_to_payload(
            [gold], dtype="bfloat16", model_id="m", request_id="req-1"
        )

    drained_job: list[PrefillJob] = []
    fetch_result: list[PrefillResult] = []

    def fetcher() -> None:
        fetch_result.append(
            remote_prefill_fetch(
                endpoint=f"127.0.0.1:{port}",
                request=PrefillRequest(
                    model_id="m", token_ids=list(range(seq_len)), request_id="req-1"
                ),
            )
        )

    fetch = threading.Thread(target=fetcher, daemon=True)
    fetch.start()
    try:
        # Test thread acts as the runner's main thread: drain the queue.
        job, event, holder = request_queue.get(timeout=5)
        drained_job.append(job)
        try:
            holder[0] = serve_one()
        finally:
            event.set()
        fetch.join(timeout=5)
        assert fetch_result, "fetcher did not return"
        result = fetch_result[0]
        assert drained_job[0].request_id == "req-1"
        assert result.total_tokens == seq_len

        dst = KVCache()
        ingest_into_mlx_cache(result, [dst])
        assert dst.offset == seq_len
        dst_k = dst.keys
        dst_v = dst.values
        gold_k = gold.keys
        gold_v = gold.values
        assert dst_k is not None and dst_v is not None
        assert gold_k is not None and gold_v is not None
        assert _equal(dst_k, gold_k)
        assert _equal(dst_v, gold_v)
    finally:
        server.stop()

    # Silence unused imports referenced only for completeness.
    _ = (
        Callable,
        KVChunk,
        Done,
        chunk_to_mlx_nhd,
        io.BytesIO,
        make_header,
        read_header,
        read_message,
        write_done,
        write_header,
    )
