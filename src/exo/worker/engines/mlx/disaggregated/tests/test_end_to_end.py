from typing import BinaryIO

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.disaggregated.protocol import Header, write_done, write_header
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated.adapter import (
    send_mlx_kv_cache,
    wire_dtype_from_cache,
    write_prefill_done,
    write_prefill_header,
    write_prefill_step,
)
from exo.worker.engines.mlx.disaggregated.client import (
    ingest_into_mlx_cache,
    remote_prefill_fetch,
    remote_prefill_stream,
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

    def resolve(job: PrefillRequest, wfile: BinaryIO) -> bool:
        _stream_cache(wfile, gold, request_id=job.request_id)
        return True

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=52417)
    try:
        result = remote_prefill_fetch(
            endpoint="127.0.0.1:52417",
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
    def resolve(_job: PrefillRequest, _wfile: BinaryIO) -> bool:
        return False

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=52418)
    try:
        with pytest.raises(RuntimeError, match="not picked up"):
            _ = remote_prefill_fetch(
                endpoint="127.0.0.1:52418",
                request=PrefillRequest(
                    model_id="test-model",
                    token_ids=[1, 2, 3],
                    request_id="never-registered",
                ),
            )
    finally:
        server.stop()


def _stream_cache_in_steps(
    wfile: BinaryIO,
    cache: KVCache,
    *,
    request_id: str,
    step_size: int,
    start_pos: int = 0,
) -> None:
    dtype = write_prefill_header(
        wfile,
        [cache],
        request_id=request_id,
        model_id="test-model",
        start_pos=start_pos,
    )
    cache_offset = int(cache.offset)
    cur = max(start_pos, 0)
    total = 0
    while cur < cache_offset:
        nxt = min(cur + step_size, cache_offset)
        sent = write_prefill_step(
            wfile, [cache], dtype=dtype, token_start=cur, token_end=nxt
        )
        total += sent
        cur = nxt
    write_prefill_done(wfile, [cache], total_tokens=total)


@pytest.mark.slow
def test_streaming_roundtrip_matches_one_shot() -> None:
    seq_len = 12
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    def resolve(job: PrefillRequest, wfile: BinaryIO) -> bool:
        _stream_cache_in_steps(wfile, gold, request_id=job.request_id, step_size=4)
        return True

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=52420)
    try:
        dst = KVCache()
        _, final_offset = remote_prefill_stream(
            endpoint="127.0.0.1:52420",
            request=PrefillRequest(
                model_id="test-model",
                token_ids=list(range(seq_len)),
                request_id="req-stream",
            ),
            caches=[dst],
        )
        assert final_offset == seq_len
        assert dst.offset == seq_len
        gold_k = gold.keys
        gold_v = gold.values
        dst_k = dst.keys
        dst_v = dst.values
        assert gold_k is not None and gold_v is not None
        assert dst_k is not None and dst_v is not None
        assert _equal(dst_k, gold_k)
        assert _equal(dst_v, gold_v)
    finally:
        server.stop()


@pytest.mark.slow
def test_streaming_roundtrip_with_start_pos() -> None:
    seq_len = 12
    start_pos = 5
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    def resolve(job: PrefillRequest, wfile: BinaryIO) -> bool:
        _stream_cache_in_steps(
            wfile, gold, request_id=job.request_id, step_size=3, start_pos=start_pos
        )
        return True

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=52421)
    try:
        dst = KVCache()
        gold_k = gold.keys
        gold_v = gold.values
        assert gold_k is not None and gold_v is not None
        dst.keys = mx.array(gold_k[:, :, :start_pos, :])
        dst.values = mx.array(gold_v[:, :, :start_pos, :])
        dst.offset = start_pos

        _, final_offset = remote_prefill_stream(
            endpoint="127.0.0.1:52421",
            request=PrefillRequest(
                model_id="test-model",
                token_ids=list(range(seq_len)),
                request_id="req-stream-prefix",
                start_pos=start_pos,
            ),
            caches=[dst],
        )
        assert final_offset == seq_len
        assert dst.offset == seq_len
        dst_k = dst.keys
        dst_v = dst.values
        assert dst_k is not None and dst_v is not None
        assert _equal(dst_k, gold_k)
        assert _equal(dst_v, gold_v)
    finally:
        server.stop()


@pytest.mark.slow
def test_server_client_roundtrip_with_start_pos() -> None:
    seq_len = 8
    start_pos = 5
    n_heads = 2
    head_dim = 4
    gold = _make_cache(seq_len, n_heads, head_dim)

    def resolve(job: PrefillRequest, wfile: BinaryIO) -> bool:
        _stream_cache(wfile, gold, request_id=job.request_id, start_pos=start_pos)
        return True

    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=52419)
    try:
        result = remote_prefill_fetch(
            endpoint="127.0.0.1:52419",
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
