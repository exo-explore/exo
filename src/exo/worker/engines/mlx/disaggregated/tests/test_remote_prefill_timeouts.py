"""How long a remote prefill fetch waits, and what it reports while it waits.

A prefill node that answers only once its prefill is done (EXO's own server does)
sends nothing until then, so the wait for the first layer of KV covers the remote
prefill itself and must not be cut by the per-receive timeout that guards a flowing
stream. The fetch still fails promptly when the node goes away, and it calls on_wait
about once a tick so the runner can report that it is alive.

Times are scaled down: the idle timeout is a fraction of a second, and so is the tick.
"""

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import BinaryIO, cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.utils.ports import random_ephemeral_port
from exo.worker.disaggregated.protocol import (
    Header,
    write_done,
    write_header,
    write_kv_chunk,
)
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated import client
from exo.worker.engines.mlx.disaggregated.client import remote_prefill_fetch
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.runner.supervisor import RUNNER_STALL_TIMEOUT

_LAYERS = 4
_TOKENS = 16
_HEADS = 2
_HEAD_DIM = 4
_BLOB = np.zeros(_TOKENS * _HEADS * _HEAD_DIM, dtype=np.uint16).tobytes()

_IDLE = 0.3
_TICK = 0.05


def _prefill_node(
    *,
    wait_before_header: float = 0.0,
    wait_before_first_layer: float = 0.0,
    wait_after_first_layer: float = 0.0,
) -> Callable[[PrefillRequest, BinaryIO], bool]:
    def resolve(_job: PrefillRequest, wfile: BinaryIO) -> bool:
        time.sleep(wait_before_header)
        write_header(wfile, Header(num_layers=_LAYERS, dtype="bfloat16"))
        wfile.flush()
        time.sleep(wait_before_first_layer)
        for layer in range(_LAYERS):
            if layer == 1:
                time.sleep(wait_after_first_layer)
            write_kv_chunk(
                wfile,
                layer_idx=layer,
                num_tokens=_TOKENS,
                n_heads=_HEADS,
                head_dim=_HEAD_DIM,
                dtype="bfloat16",
                keys=_BLOB,
                values=_BLOB,
            )
            wfile.flush()
        write_done(wfile, _TOKENS)
        wfile.flush()
        return True

    return resolve


@contextmanager
def _serving(resolve: Callable[[PrefillRequest, BinaryIO], bool]) -> Iterator[str]:
    port = random_ephemeral_port()
    server = PrefillServer(resolve=resolve, host="127.0.0.1", port=port)
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop()


def _request() -> PrefillRequest:
    return PrefillRequest(
        model_id="m", token_ids=list(range(_TOKENS)), request_id="timeouts"
    )


@pytest.fixture
def short_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "_WAIT_TICK_SECS", _TICK)


def test_the_waits_fit_the_runner_supervisor() -> None:
    # on_wait is the runner's sign of life while the fetch waits: it must come far
    # more often than the supervisor's stall timeout.
    assert client._WAIT_TICK_SECS < RUNNER_STALL_TIMEOUT / 10  # pyright: ignore[reportPrivateUsage]
    # The wait for the first layer is never shorter than the old per-receive timeout,
    # and it grows with the prompt.
    assert client.first_layer_timeout(0) >= client._SOCKET_TIMEOUT_SECS  # pyright: ignore[reportPrivateUsage]
    assert client.first_layer_timeout(32_000) > client.first_layer_timeout(2_000)


def test_keepalive_is_enabled() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        client._enable_keepalive(sock)  # pyright: ignore[reportPrivateUsage]
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(
            socket, "TCP_KEEPALIVE", None
        )
        if isinstance(idle, int):
            assert (
                sock.getsockopt(socket.IPPROTO_TCP, idle) == client._KEEPALIVE_IDLE_SECS  # pyright: ignore[reportPrivateUsage]
            )


@pytest.mark.slow
@pytest.mark.usefixtures("short_tick")
@pytest.mark.parametrize(
    "silence",
    ["before_header", "between_header_and_first_layer"],
)
def test_a_prefill_longer_than_the_idle_timeout_completes(silence: str) -> None:
    # The node computes for several idle timeouts before its first layer: before its
    # header (EXO's server, a buffered bridge) or after it.
    wait = 4 * _IDLE
    resolve = (
        _prefill_node(wait_before_header=wait)
        if silence == "before_header"
        else _prefill_node(wait_before_first_layer=wait)
    )
    waits: list[float] = []
    start = time.monotonic()
    with _serving(resolve) as endpoint:
        result = remote_prefill_fetch(
            endpoint,
            _request(),
            timeout_secs=_IDLE,
            first_layer_timeout_secs=10 * _IDLE,
            on_wait=lambda: waits.append(time.monotonic() - start),
        )
    assert result.total_tokens == _TOKENS
    assert sorted(result.kv_chunks) == list(range(_LAYERS))
    # on_wait came about once a tick all through the silence.
    marks = [0.0, *waits]
    assert marks[-1] >= wait - 2 * _TICK
    assert max(b - a for a, b in zip(marks, marks[1:], strict=False)) < 3 * _TICK


@pytest.mark.slow
@pytest.mark.usefixtures("short_tick")
def test_a_silent_prefill_node_fails_at_the_first_layer_deadline() -> None:
    deadline = 3 * _IDLE
    start = time.monotonic()
    with (
        _serving(_prefill_node(wait_before_header=10 * deadline)) as endpoint,
        pytest.raises(TimeoutError),
    ):
        remote_prefill_fetch(
            endpoint,
            _request(),
            timeout_secs=_IDLE,
            first_layer_timeout_secs=deadline,
        )
    elapsed = time.monotonic() - start
    assert deadline <= elapsed < deadline + 4 * _TICK


@pytest.mark.slow
@pytest.mark.usefixtures("short_tick")
def test_a_stream_that_stops_after_its_first_layer_fails_at_the_idle_timeout() -> None:
    # Once KV flows, the prefill is done on the node: a long silence is a stall, and
    # the generous first-layer budget no longer applies.
    start = time.monotonic()
    with (
        _serving(_prefill_node(wait_after_first_layer=20 * _IDLE)) as endpoint,
        pytest.raises(TimeoutError),
    ):
        remote_prefill_fetch(
            endpoint,
            _request(),
            timeout_secs=_IDLE,
            first_layer_timeout_secs=40 * _IDLE,
        )
    # Well short of the first-layer budget (it includes connecting and the first layer).
    elapsed = time.monotonic() - start
    assert _IDLE <= elapsed < 3 * _IDLE


@pytest.mark.slow
@pytest.mark.usefixtures("short_tick")
def test_a_closed_connection_fails_at_once() -> None:
    # A prefill node process that dies closes its connections: no deadline is waited.
    listener = socket.create_server(("127.0.0.1", 0))
    address = cast(tuple[str, int], listener.getsockname())
    port = address[1]

    def accept_and_close() -> None:
        conn = listener.accept()[0]
        conn.recv(65536)
        time.sleep(2 * _TICK)
        conn.close()

    thread = threading.Thread(target=accept_and_close, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        with pytest.raises(ConnectionError):
            remote_prefill_fetch(
                f"127.0.0.1:{port}",
                _request(),
                timeout_secs=10.0,
                first_layer_timeout_secs=10.0,
            )
    finally:
        thread.join(timeout=5)
        listener.close()
    assert time.monotonic() - start < 1.0


@pytest.mark.slow
@pytest.mark.usefixtures("short_tick")
def test_remote_prefill_reports_progress_while_the_prefill_node_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runner's progress reports are its only sign of life during a remote prefill,
    # and the supervisor stops a runner that is silent for RUNNER_STALL_TIMEOUT. A
    # buffered prefill node is silent for the whole of its prefill.
    monkeypatch.setattr(client, "_SOCKET_TIMEOUT_SECS", _IDLE)
    monkeypatch.setattr(client, "EXO_REMOTE_PREFILL_WAIT_SECONDS", 20 * _IDLE)
    wait = 4 * _IDLE
    times: list[float] = []
    reports: list[tuple[int, int]] = []
    start = time.monotonic()

    def on_progress(processed: int, total: int) -> None:
        times.append(time.monotonic() - start)
        reports.append((processed, total))

    with _serving(_prefill_node(wait_before_header=wait)) as endpoint:
        caches = [KVCache() for _ in range(_LAYERS)]
        _, num_tokens, _ = remote_prefill(
            mx.arange(_TOKENS),
            caches,
            on_progress,
            endpoint=endpoint,
            request_id="timeouts",
            model_id="m",
        )
    end = time.monotonic() - start
    assert num_tokens == _TOKENS
    assert all(c.offset == _TOKENS for c in caches)
    assert end >= wait
    marks = [0.0, *times]
    assert max(b - a for a, b in zip(marks, marks[1:], strict=False)) < 3 * _TICK
    assert all(total == _TOKENS for _, total in reports)
