"""A remote prefill reports progress while its KV stream arrives.

The runner supervisor stops a generating runner that reports nothing for
RUNNER_STALL_TIMEOUT, and during a remote prefill the progress reports are all the
runner sends. These tests use a prefill server that streams its layers slowly, as a
prefill node that streams KV while it computes does on a long or cold prefill.
"""

import time
from collections.abc import Callable
from typing import BinaryIO

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

import exo.worker.engines.mlx.generator.remote_prefill as remote_prefill_module
from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.protocol import (
    Header,
    write_done,
    write_header,
    write_kv_chunk,
)
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated import client
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.runner.supervisor import RUNNER_STALL_TIMEOUT

_LAYERS = 6
_TOKENS = 32
_HEADS = 2
_HEAD_DIM = 4
_BLOB = np.zeros(_TOKENS * _HEADS * _HEAD_DIM, dtype=np.uint16).tobytes()


def _streaming_server(
    *,
    layer_gap: float,
    chunks_per_layer: int = 1,
    wait_before_header: float = 0.0,
) -> Callable[[PrefillRequest, BinaryIO], bool]:
    """A prefill node that sends the header and layer 0 at once, then one layer per gap."""
    tokens_per_chunk = _TOKENS // chunks_per_layer

    def resolve(_job: PrefillRequest, wfile: BinaryIO) -> bool:
        time.sleep(wait_before_header)
        write_header(wfile, Header(num_layers=_LAYERS, dtype="bfloat16"))
        for layer in range(_LAYERS):
            if layer > 0:
                time.sleep(layer_gap)
            for _ in range(chunks_per_layer):
                blob = _BLOB[: tokens_per_chunk * _HEADS * _HEAD_DIM * 2]
                write_kv_chunk(
                    wfile,
                    layer_idx=layer,
                    num_tokens=tokens_per_chunk,
                    n_heads=_HEADS,
                    head_dim=_HEAD_DIM,
                    dtype="bfloat16",
                    keys=blob,
                    values=blob,
                )
            wfile.flush()
        write_done(wfile, _TOKENS)
        wfile.flush()
        return True

    return resolve


def _run(
    resolve: Callable[[PrefillRequest, BinaryIO], bool],
) -> tuple[list[float], list[tuple[int, int]], float]:
    """Remote prefill against ``resolve``: report times, reports, and the end time."""
    port = random_ephemeral_port()
    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    times: list[float] = []
    reports: list[tuple[int, int]] = []
    start = time.monotonic()

    def on_progress(processed: int, total: int) -> None:
        times.append(time.monotonic() - start)
        reports.append((processed, total))

    try:
        caches = [KVCache() for _ in range(_LAYERS)]
        _, num_tokens, _ = remote_prefill(
            mx.arange(_TOKENS),
            caches,
            on_progress,
            endpoint=f"127.0.0.1:{port}",
            request_id="progress",
            model_id="m",
        )
        end = time.monotonic() - start
        assert num_tokens == _TOKENS
        assert all(c.offset == _TOKENS for c in caches)
    finally:
        server.stop()
    return times, reports, end


def test_fetch_timeout_is_below_the_runner_stall_timeout() -> None:
    # A wait the client's socket timeout ends (the fetch fails, the caller prefills
    # locally) must end before the supervisor takes the silent runner for a stuck one.
    assert client._SOCKET_TIMEOUT_SECS < RUNNER_STALL_TIMEOUT  # pyright: ignore[reportPrivateUsage]
    assert remote_prefill_module.PROGRESS_INTERVAL_SECONDS < RUNNER_STALL_TIMEOUT


@pytest.mark.slow
def test_a_slow_stream_reports_progress_as_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gap = 0.15
    monkeypatch.setattr(remote_prefill_module, "PROGRESS_INTERVAL_SECONDS", gap / 3)
    times, reports, end = _run(_streaming_server(layer_gap=gap))

    # The stream takes several gaps, and no stretch of it goes unreported for much
    # longer than one gap: from the start, between reports, and up to the return.
    assert end > (_LAYERS - 1) * gap
    marks = [0.0, *times, end]
    silences = [b - a for a, b in zip(marks, marks[1:], strict=False)]
    assert max(silences) < 2 * gap, silences
    assert len(reports) >= _LAYERS


@pytest.mark.slow
@pytest.mark.parametrize("chunks_per_layer", [1, 2])
def test_progress_counts_tokens_up_to_the_prompt_length(
    monkeypatch: pytest.MonkeyPatch, chunks_per_layer: int
) -> None:
    monkeypatch.setattr(remote_prefill_module, "PROGRESS_INTERVAL_SECONDS", 0.0)
    _, reports, _ = _run(
        _streaming_server(layer_gap=0.0, chunks_per_layer=chunks_per_layer)
    )

    processed = [p for p, _ in reports]
    assert all(total == _TOKENS for _, total in reports)
    assert processed == sorted(processed)
    assert processed[-1] == _TOKENS
    # Before connecting, at the header, then once per chunk.
    assert len(reports) == 2 + _LAYERS * chunks_per_layer


@pytest.mark.slow
def test_progress_reports_are_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_prefill_module, "PROGRESS_INTERVAL_SECONDS", 60.0)
    _, reports, _ = _run(_streaming_server(layer_gap=0.0))

    # Before connecting and at the header; every chunk falls inside the interval.
    assert reports == [(0, _TOKENS), (0, _TOKENS)]


@pytest.mark.slow
def test_a_silent_prefill_node_fails_the_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing arrives for longer than the socket timeout: the fetch raises, which is
    # where BatchGenerator.submit and mlx_generate fall back to local prefill.
    timeout = 0.2
    fetch = client.remote_prefill_fetch
    defaults = fetch.__defaults__
    assert defaults is not None
    monkeypatch.setattr(fetch, "__defaults__", (*defaults[:2], timeout, *defaults[3:]))

    # The node would answer in full after three timeouts, so only the timeout raises.
    with pytest.raises(TimeoutError):
        _run(_streaming_server(layer_gap=0.0, wait_before_header=3 * timeout))
