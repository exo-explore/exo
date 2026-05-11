"""Direct TCP socket transport for the asymmetric drafter wire.

The original drafter wire (:mod:`remote_drafter`) carries small uint32
arrays via ``mx.distributed.send/recv`` over the parent
``mx.distributed.Group``. That design forces the drafter rank to be a
member of the parent group, which in turn requires
``mx.distributed.Group.split`` so target ranks can run TP/PP collectives
without dragging the drafter in. JACCL and ring backends do not
implement ``split`` on Apple Silicon, so the V1 asymmetric path was
limited to a single target rank.

This module breaks that coupling. The drafter rank no longer joins
``mx.distributed`` at all. Instead, target rank 0 binds a TCP server
socket at instance bootstrap time, the drafter dials it, and the same
wire frames flow over that connection. The target's
``mx.distributed.Group`` therefore contains only target ranks and is
free to do whatever TP/PP work it needs without ``Group.split``.

Wire frames are length-implicit (every op type has a known fixed shape;
``OP_PREFILL`` carries a variable-length token array whose length is
announced in the preceding command frame's ``num_forwards`` slot). Each
uint32 is serialised little-endian, matching mlx_lm's on-device layout
for ``mx.uint32``.

Threading model: both the target rank's ``RemoteTransport`` and the
drafter rank's serve loop run wire ops serially on a single thread (the
target uses a single-worker ``ThreadPoolExecutor``; the drafter loops
synchronously). Concurrency is multiplexed via session ids, not via
multiple sockets, so a single TCP connection per asymmetric instance is
sufficient and avoids mid-flight reordering.
"""

from __future__ import annotations

import contextlib
import socket
import struct
import time
from typing import Final

_HEADER_FORMAT: Final[str] = "<I"
"""Length prefix for variable-length payloads.

Used only for OP_PREFILL's prompt-token tail. Fixed-shape frames don't
need a header because both sides know the shape statically."""


def send_uint32_frame(sock: socket.socket, values: list[int]) -> None:
    """Send a fixed-length uint32 frame over ``sock``.

    Caller must guarantee both peers know the frame length statically;
    no length prefix is sent. Suitable for command/ack/drafts frames.
    """
    if not all(0 <= v <= 0xFFFFFFFF for v in values):
        raise ValueError(f"frame contains non-uint32 values: {values}")
    payload = struct.pack(f"<{len(values)}I", *values)
    sock.sendall(payload)


def recv_uint32_frame(sock: socket.socket, count: int) -> list[int]:
    """Receive ``count`` uint32 ints over ``sock`` (no length prefix).

    Blocks until ``count * 4`` bytes have been received, raising
    :class:`ConnectionError` if the peer closes mid-frame.
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
                f"drafter wire closed mid-frame (received {received}/{needed} bytes)"
            )
        received += chunk
    unpacked = struct.unpack(f"<{count}I", bytes(buf))
    return list(unpacked)


def send_variable_uint32_payload(sock: socket.socket, values: list[int]) -> None:
    """Send a length-prefixed uint32 payload (4-byte header + values).

    Used for OP_PREFILL's prompt-token tail when the size isn't carried
    in the preceding command frame's slot.
    """
    if not all(0 <= v <= 0xFFFFFFFF for v in values):
        raise ValueError("variable payload contains non-uint32 values")
    header = struct.pack(_HEADER_FORMAT, len(values))
    sock.sendall(header)
    if values:
        sock.sendall(struct.pack(f"<{len(values)}I", *values))


def bind_target_listener(host: str, port: int, *, backlog: int = 1) -> socket.socket:
    """Open and listen on ``(host, port)`` for the drafter's incoming dial.

    Address family is resolved via ``getaddrinfo`` rather than
    hard-coded to ``AF_INET``: placement-time
    ``DrafterPlacement.drafter_socket_host`` can pick a non-IPv4
    address (Tailscale ULA, link-local IPv6, etc.) on IPv6-only
    links, and a hard-coded IPv4 listener would refuse the bind tuple
    or accept connections only on IPv4 while the drafter dials IPv6 --
    the asymmetric instance would never reach warmup.

    When binding to an IPv6 wildcard (``"::"``), dual-stack mode is
    forced via ``IPV6_V6ONLY=0`` so a single listener accepts both
    IPv6 and IPv4 connects on platforms where dual-stack is
    off-by-default (Linux).

    Bound with ``SO_REUSEADDR`` so a previous instance teardown that
    left the port in TIME_WAIT does not block reclaim. Caller is
    responsible for ``accept()`` and ``close()``.

    Codex P1 (PR #20, placement.py:711): the ``port`` argument arrives
    via :class:`DrafterPlacement.drafter_socket_port`, picked by the
    master via :func:`exo.utils.ports.random_ephemeral_port`. As of
    that helper's PR #20 round-(N+12) rewrite, the picked port is
    kernel-vetted-free on the master's host -- which is the same host
    as ``bind_target_listener``'s caller in the dominant single-machine
    deploy. For cross-machine deploys the master cannot vet the
    target's port, so ``EADDRINUSE`` here is still possible; in that
    case the listener raises ``OSError`` with a self-describing
    message that names the placement-supplied port so an operator can
    correlate the placement event with the bind failure immediately
    rather than chasing a generic ``Address already in use``.
    """
    # ``AI_PASSIVE`` lets ``host=""``/wildcard resolve to the
    # right family-specific wildcard (``0.0.0.0`` / ``::``);
    # literal addresses and hostnames also resolve correctly.
    addrinfo = socket.getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
        flags=socket.AI_PASSIVE,
    )
    if not addrinfo:
        raise ValueError(
            f"unable to resolve bind address for drafter listener: {host}:{port}"
        )
    family, socktype, proto, _canonname, sockaddr = addrinfo[0]
    listener = socket.socket(family, socktype, proto)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        # Force dual-stack so an IPv6 wildcard accepts IPv4-mapped
        # connects (``::ffff:a.b.c.d``); on Linux this is off by
        # default, on BSDs it is on. Force-enable for cross-platform
        # parity. ``IPV6_V6ONLY`` is unavailable on some platforms
        # (notably Windows pre-Vista); in that case the listener
        # still works for IPv6-only traffic, which is an acceptable
        # degradation.
        with contextlib.suppress(OSError, AttributeError):
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    try:
        listener.bind(sockaddr)
    except OSError as bind_error:
        # Wrap the kernel's terse "Address already in use" / "Permission
        # denied" text with the placement metadata so operators can
        # correlate the failure with the originating
        # :class:`DrafterPlacement` event without grepping logs.
        listener.close()
        raise OSError(
            bind_error.errno,
            f"failed to bind drafter listener at {host}:{port} "
            f"({bind_error.strerror or bind_error}); the placement-"
            f"selected port may already be in use on this host. The "
            f"master picks ports via ``random_ephemeral_port`` which is "
            f"kernel-vetted on the master's host but not across hosts; "
            f"in cross-machine deploys this is a benign retry-able "
            f"condition (the runner will be re-placed with a fresh "
            f"port).",
        ) from bind_error
    listener.listen(backlog)
    return listener


def accept_drafter(
    listener: socket.socket,
    *,
    timeout_seconds: float = 60.0,
) -> socket.socket:
    """Block on ``listener.accept`` for the drafter's incoming connection.

    The drafter dials soon after target rank 0 reaches its
    ``ConnectToGroup`` step, so a generous default timeout (60s) covers
    drafter-side weight loading and warmup without spinning. ``TCP_NODELAY``
    is set on the accepted socket because every wire op is a small
    request/reply round trip; Nagle would add ~40ms of latency per op
    while batching tiny frames.
    """
    listener.settimeout(timeout_seconds)
    try:
        accepted = listener.accept()
    finally:
        listener.settimeout(None)
    conn: socket.socket = accepted[0]
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def dial_target(
    host: str,
    port: int,
    *,
    total_timeout_seconds: float = 120.0,
    initial_backoff_seconds: float = 0.5,
) -> socket.socket:
    """Dial ``(host, port)`` with exponential backoff until connected.

    Used by the drafter rank to reach target rank 0's listener. Target
    rank 0 binds inside its ``ConnectToGroup`` step, which races with
    the drafter rank's bootstrap; the drafter therefore retries until
    the listener is up or the deadline expires. Backoff caps at 5s
    between attempts so we don't sleep through a transient binding
    hiccup.

    Codex P2 (PR #20 round-(N+13), drafter_socket.py:195): each
    ``socket.create_connection`` attempt uses the *remaining* time
    until the deadline (clamped to a sensible upper bound) instead
    of a fixed ``min(10.0, total_timeout_seconds)``. Pre-fix the
    final attempt could block past the configured total timeout
    (plus the backoff sleep already accounted for), so asymmetric
    runner failure detection was delayed and the connect phase
    kept hanging well after the caller's deadline -- despite the
    error message claiming the failure occurred *within*
    ``total_timeout_seconds``.
    """
    deadline = time.monotonic() + total_timeout_seconds
    backoff = initial_backoff_seconds
    last_error: BaseException | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        # Cap any single attempt at 10s so a slow / black-hole TCP
        # SYN doesn't burn the entire deadline on the first try; if
        # the deadline is already shorter than 10s, honour it.
        attempt_timeout = min(10.0, remaining)
        try:
            conn = socket.create_connection((host, port), timeout=attempt_timeout)
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            last_error = exc
            # Don't sleep past the deadline; the next loop iteration
            # would just exit immediately, and we want the failure
            # error to surface as close to the deadline as possible.
            sleep_for = min(backoff, max(deadline - time.monotonic(), 0.0))
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            backoff = min(backoff * 2.0, 5.0)
    raise ConnectionError(
        f"drafter could not reach target rank 0 at {host}:{port} "
        f"within {total_timeout_seconds:.0f}s "
        f"(last error: {last_error!r})"
    )


__all__ = [
    "accept_drafter",
    "bind_target_listener",
    "dial_target",
    "recv_uint32_frame",
    "send_uint32_frame",
    "send_variable_uint32_payload",
]
