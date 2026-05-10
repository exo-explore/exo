"""Direct TCP socket transport for target-rank-to-peer broadcasts.

Mirrors :mod:`drafter_socket` but for inter-target-rank communication
during the speculative-decode hot path. The hot path needs to broadcast
small int32 buffers from target rank 0 to every other target rank
(drafts on the way in, sampled tokens on the way out). The original
implementation rode :func:`mx.distributed.all_sum` and later
:func:`mx.distributed.send` / :func:`recv` over the same target group
that runs the model's tensor-parallel ``all_sum`` collectives.

That coupling is the bug: the JACCL backend interleaves the int32
broadcast with the float32 TP all-reduce on the same wire, occasionally
handing back logits memory in place of the requested int32 buffer.
Symptom is a deterministic out-of-vocabulary token id (the bit pattern
of a float32 logit reinterpreted as int32) emerging on the receiving
peer rank a few hundred milliseconds into generation.

Fix: lift the int32 broadcasts off ``mx.distributed`` entirely. Target
rank 0 binds a TCP listener at instance bootstrap; every other target
rank dials in once and reuses the connection for the lifetime of the
runner. The wire is fundamentally separate from JACCL, so the model's
TP collectives and the spec-decode broadcasts can never collide.

Wire frames are fixed-length little-endian int32 sequences, matching
:mod:`drafter_socket` for consistency. Unlike the drafter wire, every
spec-decode broadcast has a known shape (``k + 1`` ints), so no
length-prefixed payloads are needed.

Threading model: the spec-decode loop is single-threaded per runner;
target rank 0's broadcast issues one ``sendall`` per peer, peers issue
one ``recv_into`` per round. No multiplexing, no out-of-order frames.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import Final

_INT32_MIN: Final[int] = -(1 << 31)
_INT32_MAX: Final[int] = (1 << 31) - 1


def send_int32_frame(sock: socket.socket, values: list[int]) -> None:
    """Send a fixed-length signed int32 frame over ``sock``.

    The spec-decode loop only ever broadcasts non-negative token ids
    and length prefixes today, but signed int32 covers both that case
    and any future sentinel (e.g. -1 for "end of stream") without
    revisiting the wire format. Callers must guarantee the peer
    expects exactly ``len(values)`` ints; no length header is sent.
    """
    for index, value in enumerate(values):
        if value < _INT32_MIN or value > _INT32_MAX:
            raise ValueError(
                f"target-peer frame value at index {index}={value} is out of "
                f"int32 range [{_INT32_MIN}, {_INT32_MAX}]"
            )
    payload = struct.pack(f"<{len(values)}i", *values)
    sock.sendall(payload)


def recv_int32_frame(sock: socket.socket, count: int) -> list[int]:
    """Receive ``count`` signed int32 ints over ``sock`` (no length prefix).

    Blocks until ``count * 4`` bytes have arrived, raising
    :class:`ConnectionError` if the peer closes mid-frame so the
    spec-decode loop surfaces a typed wire failure rather than a
    silent truncated buffer.
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    needed = count * 4
    buf = bytearray(needed)
    view = memoryview(buf)
    received = 0
    while received < needed:
        chunk = sock.recv_into(view[received:], needed - received)
        if chunk == 0:
            raise ConnectionError(
                f"target-peer wire closed mid-frame "
                f"(received {received}/{needed} bytes)"
            )
        received += chunk
    unpacked = struct.unpack(f"<{count}i", bytes(buf))
    return list(unpacked)


def bind_target_peer_listener(host: str, port: int, *, backlog: int) -> socket.socket:
    """Open and listen on ``(host, port)`` for peer target ranks to dial in.

    ``backlog`` is the expected number of dialing peers
    (``target_world_size - 1``). ``SO_REUSEADDR`` is set so a stale
    TIME_WAIT socket from a previous instance teardown does not block
    rebind. Caller owns ``accept()`` (see :func:`accept_target_peers`)
    and ``close()``.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(backlog)
    return listener


def accept_target_peers(
    listener: socket.socket,
    *,
    expected_peers: int,
    timeout_seconds: float,
) -> list[socket.socket]:
    """Accept exactly ``expected_peers`` incoming target-peer connections.

    Order of acceptance is not significant for the wire protocol --
    rank 0 issues one ``sendall`` per accepted socket per broadcast,
    independent of which peer rank ended up where in the list. Callers
    that need rank-indexed access (none in the current spec-decode
    loop) must perform their own handshake on top of the returned
    sockets.

    ``TCP_NODELAY`` is set on every accepted socket. Each broadcast is
    a 24-to-200-byte int32 frame followed by a long pause (the
    verifier's TP forward pass), so Nagle would add the full 40ms
    delayed-ack timeout to every round. Disabling Nagle drops that to
    sub-millisecond on Thunderbolt RDMA.
    """
    if expected_peers <= 0:
        raise ValueError(
            f"accept_target_peers needs expected_peers >= 1, got {expected_peers}"
        )
    listener.settimeout(timeout_seconds)
    accepted: list[socket.socket] = []
    try:
        for _ in range(expected_peers):
            accept_result: tuple[socket.socket, object] = listener.accept()
            conn: socket.socket = accept_result[0]
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            accepted.append(conn)
    finally:
        listener.settimeout(None)
    return accepted


def dial_target_zero(
    host: str,
    port: int,
    *,
    total_timeout_seconds: float,
    initial_backoff_seconds: float = 0.5,
) -> socket.socket:
    """Dial target rank 0 from a peer target rank, retrying until success.

    Target rank 0 binds inside :func:`initialize_mlx` after
    ``mlx_distributed_init`` returns; peers dial during the same
    bootstrap step, so the listener may not yet be up when the first
    dial attempt fires. Exponential backoff (capped at 5s) covers the
    bind / accept race without spinning. Failure after
    ``total_timeout_seconds`` raises :class:`ConnectionError`, which
    the runner surfaces as a connect-task failure so the cluster does
    not sit silently wedged.
    """
    deadline = time.monotonic() + total_timeout_seconds
    backoff = initial_backoff_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            conn = socket.create_connection(
                (host, port), timeout=min(10.0, total_timeout_seconds)
            )
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
    raise ConnectionError(
        f"target peer could not reach target rank 0 at {host}:{port} "
        f"within {total_timeout_seconds:.0f}s (last error: {last_error!r})"
    )


__all__ = [
    "accept_target_peers",
    "bind_target_peer_listener",
    "dial_target_zero",
    "recv_int32_frame",
    "send_int32_frame",
]
