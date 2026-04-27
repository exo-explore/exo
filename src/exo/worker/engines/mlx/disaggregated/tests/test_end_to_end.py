import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.disaggregated.adapter import serialize_mlx_cache_to_payload
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillRequest,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.disaggregated.server import (
    PrefillJob,
    PrefillPayloadLookup,
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
def test_server_client_roundtrip() -> None:
    seq_len = 5
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    lookup = PrefillPayloadLookup()
    payload = serialize_mlx_cache_to_payload(
        [gold], dtype="bfloat16", model_id="test-model", request_id="req-1"
    )
    lookup.register("req-1", payload)

    def resolve(job: PrefillJob) -> bytes | None:
        return lookup.pop(job.request_id)

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=0)
    port = server.start()
    try:
        result = remote_prefill_fetch(
            endpoint=f"127.0.0.1:{port}",
            request=PrefillRequest(
                model_id="test-model",
                token_ids=list(range(seq_len)),
                request_id="req-1",
            ),
        )
        assert result.total_tokens == seq_len
        assert 0 in result.kv_chunks

        dst = KVCache()
        final_offset = ingest_into_mlx_cache(result, [dst])
        assert final_offset == seq_len
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


@pytest.mark.slow
def test_server_reports_unknown_request() -> None:
    lookup = PrefillPayloadLookup()

    def resolve(job: PrefillJob) -> bytes | None:
        return lookup.pop(job.request_id)

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=0)
    port = server.start()
    try:
        with pytest.raises(RuntimeError, match="No payload ready"):
            _ = remote_prefill_fetch(
                endpoint=f"127.0.0.1:{port}",
                request=PrefillRequest(
                    model_id="test-model",
                    token_ids=[1, 2, 3],
                    request_id="never-registered",
                ),
            )
    finally:
        server.stop()
