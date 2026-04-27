from typing import BinaryIO

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.disaggregated.protocol import Header, write_done, write_header
from exo.worker.disaggregated.server import PrefillJob, PrefillServer
from exo.worker.engines.mlx.disaggregated.adapter import (
    send_mlx_kv_cache,
    wire_dtype_from_cache,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillRequest,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
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
    with mx.stream(mx.Device(mx.cpu)):
        cache.keys = (
            mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
        ).astype(mx.bfloat16)
        cache.values = (
            mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
        ).astype(mx.bfloat16)
        mx.eval(cache.keys, cache.values)
    cache.offset = seq_len
    return cache


def _stream_cache(
    wfile: BinaryIO, cache: KVCache, *, request_id: str, start_pos: int = 0
) -> None:
    dtype = wire_dtype_from_cache([cache])
    write_header(
        wfile,
        Header(
            request_id=request_id,
            model_id="test-model",
            num_layers=1,
            dtype=dtype,
            start_pos=start_pos,
        ),
    )
    tokens_sent = send_mlx_kv_cache(wfile, [cache], dtype=dtype, start_pos=start_pos)
    write_done(wfile, tokens_sent)
    wfile.flush()


@pytest.mark.slow
def test_server_client_roundtrip() -> None:
    seq_len = 5
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    def resolve(job: PrefillJob, wfile: BinaryIO) -> bool:
        _stream_cache(wfile, gold, request_id=job.request_id)
        return True

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
def test_server_reports_pickup_failure() -> None:
    def resolve(_job: PrefillJob, _wfile: BinaryIO) -> bool:
        return False

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=0)
    port = server.start()
    try:
        with pytest.raises(RuntimeError, match="not picked up"):
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


@pytest.mark.slow
def test_server_client_roundtrip_with_start_pos() -> None:
    seq_len = 8
    start_pos = 5
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    def resolve(job: PrefillJob, wfile: BinaryIO) -> bool:
        _stream_cache(wfile, gold, request_id=job.request_id, start_pos=start_pos)
        return True

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=0)
    port = server.start()
    try:
        result = remote_prefill_fetch(
            endpoint=f"127.0.0.1:{port}",
            request=PrefillRequest(
                model_id="test-model",
                token_ids=list(range(seq_len)),
                request_id="req-1",
                start_pos=start_pos,
            ),
        )
        assert result.total_tokens == seq_len - start_pos
        assert result.header.start_pos == start_pos

        dst = KVCache()
        gold_k = gold.keys
        gold_v = gold.values
        assert gold_k is not None and gold_v is not None
        dst.keys = mx.array(gold_k[:, :, :start_pos, :])
        dst.values = mx.array(gold_v[:, :, :start_pos, :])
        dst.offset = start_pos

        final_offset = ingest_into_mlx_cache(result, [dst], start_pos=start_pos)
        assert final_offset == seq_len
        assert dst.offset == seq_len
        dst_k = dst.keys
        dst_v = dst.values
        assert dst_k is not None and dst_v is not None
        assert _equal(dst_k, gold_k)
        assert _equal(dst_v, gold_v)
    finally:
        server.stop()
