import contextlib
import io
import socket
import time
from collections import defaultdict
from collections.abc import Buffer, Callable
from dataclasses import dataclass, field
from typing import BinaryIO, cast, final, override

import mlx.core as mx
from loguru import logger
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.shared.constants import (
    EXO_REMOTE_PREFILL_WAIT_SECONDS,
    EXO_REMOTE_PREFILL_WAIT_SECONDS_PER_1K_TOKENS,
)
from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    Header,
    KVChunk,
    TensorBlob,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest, write_request
from exo.worker.engines.mlx.disaggregated.adapter import (
    chunk_to_mlx_nhd,
    inject_arrays_cache,
    inject_kv_chunk,
    inject_rotating_kv_chunk,
)

# Once the prefill node has started sending KV, the fetch fails if no byte arrives for
# this long (the caller then prefills locally).
_SOCKET_TIMEOUT_SECS = 60
# Connecting to the prefill node: it accepts at once or it is not there.
_CONNECT_TIMEOUT_SECS = 10.0
# While it waits for bytes, the fetch wakes this often to call on_wait and check its
# deadlines, so the caller can show it is alive (the runner supervisor stops a runner
# that reports nothing for RUNNER_STALL_TIMEOUT).
_WAIT_TICK_SECS = 1.0
# TCP keepalive on the connection: an idle connection to a host that has gone away
# fails after about IDLE + INTERVAL * COUNT seconds, well inside the wait for the
# first layer. A prefill node process that dies closes the connection at once.
_KEEPALIVE_IDLE_SECS = 10
_KEEPALIVE_INTERVAL_SECS = 5
_KEEPALIVE_COUNT = 3
_RECV_BUFFER_BYTES = 4 * 1024 * 1024
_READ_BUFFER_BYTES = 256 * 1024


def first_layer_timeout(num_tokens: int) -> float:
    """How long a remote prefill of ``num_tokens`` may take until its first layer of KV.

    A prefill node may send its header and KV only once its prefill is done (EXO's own
    server does), so this covers the prefill itself and grows with the prompt.
    """
    return (
        EXO_REMOTE_PREFILL_WAIT_SECONDS
        + EXO_REMOTE_PREFILL_WAIT_SECONDS_PER_1K_TOKENS * max(num_tokens, 0) / 1000
    )


def _enable_keepalive(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # Linux calls the idle time TCP_KEEPIDLE, macOS TCP_KEEPALIVE.
    idle_option = cast(
        int | None,
        getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None),
    )
    for option, value in (
        (idle_option, _KEEPALIVE_IDLE_SECS),
        (
            cast(int | None, getattr(socket, "TCP_KEEPINTVL", None)),
            _KEEPALIVE_INTERVAL_SECS,
        ),
        (cast(int | None, getattr(socket, "TCP_KEEPCNT", None)), _KEEPALIVE_COUNT),
    ):
        if option is not None:
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_TCP, option, value)


@final
class _PrefillSocketIO(io.RawIOBase):
    """Reads the prefill stream with two deadlines and a heartbeat.

    Until the first layer of KV arrives, the prefill node may be computing and send
    nothing at all, so the only bound is ``first_layer_deadline`` (absolute, from the
    request). After that, the stream fails when no byte arrives for ``idle_timeout``.
    While waiting, it wakes every ``tick`` seconds; ``on_wait`` is called at most once
    per tick for as long as the read lasts, whether or not bytes arrive.

    A socket timeout cannot do this on its own: once a socket file has timed out, it
    refuses every later read.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        first_layer_deadline: float,
        idle_timeout: float,
        tick: float,
        on_wait: Callable[[], None] | None,
    ) -> None:
        super().__init__()
        self._sock = sock
        self._first_layer_deadline: float | None = first_layer_deadline
        self._idle_timeout = idle_timeout
        self._tick = tick
        self._on_wait = on_wait
        now = time.monotonic()
        self._last_data = now
        self._last_wait = now
        sock.settimeout(tick)

    def first_layer_arrived(self) -> None:
        self._first_layer_deadline = None
        self._last_data = time.monotonic()

    @override
    def readable(self) -> bool:
        return True

    @override
    def readinto(self, buffer: Buffer) -> int:
        while True:
            try:
                received = self._sock.recv_into(buffer)
            except TimeoutError:
                received = None
            now = time.monotonic()
            if self._on_wait is not None and now - self._last_wait >= self._tick:
                self._last_wait = now
                self._on_wait()
            if received is not None:
                self._last_data = now
                return received
            if self._first_layer_deadline is not None:
                if now >= self._first_layer_deadline:
                    raise TimeoutError(
                        "Remote prefill: no KV from the prefill node in time"
                    )
            elif now - self._last_data >= self._idle_timeout:
                raise TimeoutError(
                    f"Remote prefill: nothing received for {self._idle_timeout:.0f}s"
                )


@dataclass
class PrefillResult:
    header: Header
    kv_chunks: dict[int, list[KVChunk]] = field(
        default_factory=dict[int, list[KVChunk]]
    )
    arrays: dict[int, list[TensorBlob]] = field(
        default_factory=dict[int, list[TensorBlob]]
    )
    total_tokens: int = 0


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    raise ValueError(f"Invalid endpoint {endpoint}")


def remote_prefill_fetch(
    endpoint: str,
    request: PrefillRequest,
    on_header: Callable[[Header], None] | None = None,
    on_kv_chunk: Callable[[KVChunk, int], None] | None = None,
    timeout_secs: float | None = None,
    *,
    on_wait: Callable[[], None] | None = None,
    first_layer_timeout_secs: float | None = None,
    connect_timeout_secs: float = _CONNECT_TIMEOUT_SECS,
) -> PrefillResult:
    """Fetch a remote prefill.

    ``timeout_secs`` (default ``_SOCKET_TIMEOUT_SECS``) bounds each wait for bytes once
    KV has started to arrive.
    ``first_layer_timeout_secs`` (default: ``first_layer_timeout`` of the tokens to
    prefill) bounds the time from the request to the first layer of KV, during which
    the prefill node may be computing and silent. ``on_wait`` is called about once per
    ``_WAIT_TICK_SECS`` while the fetch waits for the stream. Raises TimeoutError,
    ConnectionError or OSError when the stream fails; callers then prefill locally.
    """
    host, port = _parse_endpoint(endpoint)
    logger.info(
        f"Connecting to prefill server at {host}:{port} "
        f"({len(request.token_ids)} tokens, start_pos={request.start_pos})"
    )

    if timeout_secs is None:
        timeout_secs = _SOCKET_TIMEOUT_SECS
    if first_layer_timeout_secs is None:
        first_layer_timeout_secs = first_layer_timeout(
            len(request.token_ids) - request.start_pos
        )
    started = time.monotonic()
    sock = socket.create_connection((host, port), timeout=connect_timeout_secs)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER_BYTES)
    _enable_keepalive(sock)
    try:
        sock.settimeout(timeout_secs)
        wfile = sock.makefile("wb", buffering=256 * 1024)
        wstream: BinaryIO = cast(BinaryIO, cast(object, wfile))
        write_request(wstream, request)

        raw = _PrefillSocketIO(
            sock,
            first_layer_deadline=started + first_layer_timeout_secs,
            idle_timeout=timeout_secs,
            tick=_WAIT_TICK_SECS,
            on_wait=on_wait,
        )
        stream: BinaryIO = cast(
            BinaryIO, cast(object, io.BufferedReader(raw, _READ_BUFFER_BYTES))
        )

        header = read_header(stream)
        if on_header is not None:
            on_header(header)

        result = PrefillResult(header=header)
        kv_by_layer: dict[int, list[KVChunk]] = defaultdict(list)
        chunks_received = 0

        while True:
            msg = read_message(stream)
            # The prefill is done on the node (or it failed): from here on, a silent
            # stream is a stalled one.
            raw.first_layer_arrived()
            if msg is None:
                break
            if isinstance(msg, KVChunk):
                kv_by_layer[msg.layer_idx].append(msg)
                chunks_received += 1
                if on_kv_chunk is not None:
                    on_kv_chunk(msg, chunks_received)
            elif isinstance(msg, ArraysState):
                result.arrays[msg.layer_idx] = msg.arrays
            elif isinstance(msg, Done):
                result.total_tokens = msg.total_tokens
                break
            else:
                raise RuntimeError(f"Prefill server error [{msg.code}]: {msg.message}")

        result.kv_chunks = dict(kv_by_layer)
        return result
    finally:
        sock.close()


def ingest_into_mlx_cache(
    result: PrefillResult,
    caches: list[KVCache | RotatingKVCache | ArraysCache],
    *,
    start_pos: int = 0,
) -> int:
    max_received = max(
        (sum(c.num_tokens for c in chunks) for chunks in result.kv_chunks.values()),
        default=0,
    )
    final_offset = start_pos + max_received

    for i, cache in enumerate(caches):
        if i in result.kv_chunks:
            chunks = result.kv_chunks[i]
            if len(chunks) == 1:
                k_nhd, v_nhd = chunk_to_mlx_nhd(chunks[0])
            else:
                decoded = [chunk_to_mlx_nhd(c) for c in chunks]
                k_nhd = mx.concatenate([k for k, _ in decoded], axis=0)
                v_nhd = mx.concatenate([v for _, v in decoded], axis=0)

            if isinstance(cache, RotatingKVCache):
                inject_rotating_kv_chunk(cache, k_nhd, v_nhd, final_offset)
            elif isinstance(cache, KVCache):
                if start_pos > 0:
                    inject_kv_chunk(
                        cache,
                        k_nhd,
                        v_nhd,
                        final_offset,
                        start_pos=start_pos,
                        existing_k=cache.keys,
                        existing_v=cache.values,
                    )
                else:
                    inject_kv_chunk(cache, k_nhd, v_nhd, final_offset)

        if i in result.arrays and isinstance(cache, ArraysCache):
            inject_arrays_cache(cache, result.arrays[i])

    return final_offset
