"""Tests for :mod:`remote_drafter` -- wire protocol + transport behaviour.

The asymmetric drafter wire is a plain TCP socket under the v3+ design;
unit tests use ``socket.socketpair()`` to exercise both sides of the
protocol end-to-end without an MLX backend or extra processes. End-to-
end correctness against a real cluster is exercised by the multi-host
benchmark runs, not in unit tests.
"""

from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Iterator

import pytest

from exo.worker.engines.mlx.generator.remote_drafter import (
    ACK_FRAME_SIZE,
    ACK_OK,
    COMMAND_FRAME_SIZE,
    OP_END_SESSION,
    OP_FORWARD,
    OP_PREFILL,
    OP_SHUTDOWN,
    OP_TRIM_CACHE,
    SESSION_ID_NONE,
    RemoteTransport,
    _build_command_frame,  # type: ignore[reportPrivateUsage]
    _decode_command_frame,  # type: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Wire protocol: command frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "inputs", "num_forwards", "trim_amount", "session_id", "target_buf"),
    [
        (OP_FORWARD, [42], 4, 0, 0, 5),
        (OP_FORWARD, [10, 20], 5, 0, 7, 6),
        (OP_TRIM_CACHE, [], 0, 7, 3, 5),
        (OP_SHUTDOWN, [], 0, 0, SESSION_ID_NONE, 5),
        (OP_PREFILL, [], 1024, 0, 1, 5),
        (OP_PREFILL, [], 0, 0, 0, 5),
        (OP_END_SESSION, [], 0, 0, 42, 5),
        (OP_FORWARD, [1], 2, 0, 0xFFFFFFFE, 9),
    ],
)
def test_command_frame_round_trip(
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
    session_id: int,
    target_buf: int,
) -> None:
    """Every command shape we send must round-trip through encode + decode."""
    flat = _build_command_frame(
        op=op,
        inputs=inputs,
        num_forwards=num_forwards,
        trim_amount=trim_amount,
        session_id=session_id,
        target_drafts_buffer_size=target_buf,
    )
    assert len(flat) == COMMAND_FRAME_SIZE

    (
        decoded_op,
        decoded_inputs,
        decoded_num_forwards,
        decoded_trim,
        decoded_sid,
        decoded_target_buf,
    ) = _decode_command_frame(flat)
    assert decoded_op == op
    assert decoded_inputs == inputs
    assert decoded_num_forwards == num_forwards
    assert decoded_trim == trim_amount
    assert decoded_sid == session_id
    assert decoded_target_buf == target_buf


def test_command_frame_rejects_long_inputs() -> None:
    with pytest.raises(ValueError, match=r"inputs length must be in \[0, 2\]"):
        _build_command_frame(
            op=OP_FORWARD,
            inputs=[1, 2, 3],
            num_forwards=4,
            trim_amount=0,
            session_id=0,
            target_drafts_buffer_size=5,
        )


def test_command_frame_rejects_session_id_out_of_uint32_range() -> None:
    with pytest.raises(ValueError, match=r"session_id must fit in uint32"):
        _build_command_frame(
            op=OP_FORWARD,
            inputs=[1],
            num_forwards=2,
            trim_amount=0,
            session_id=2**33,
            target_drafts_buffer_size=5,
        )


def test_command_frame_rejects_target_buffer_out_of_uint32_range() -> None:
    with pytest.raises(
        ValueError, match=r"target_drafts_buffer_size must fit in uint32"
    ):
        _build_command_frame(
            op=OP_FORWARD,
            inputs=[1],
            num_forwards=2,
            trim_amount=0,
            session_id=0,
            target_drafts_buffer_size=2**33,
        )


def test_decode_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match=r"expected 9"):
        _decode_command_frame([0, 0, 0])


# ---------------------------------------------------------------------------
# Helpers for socketpair-based wire tests
# ---------------------------------------------------------------------------


def _socket_pair() -> tuple[socket.socket, socket.socket]:
    """Return ``(target_side, drafter_side)`` connected unix sockets."""
    target_side, drafter_side = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    target_side.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    return target_side, drafter_side


def _read_uint32s(sock: socket.socket, count: int) -> list[int]:
    needed = count * 4
    buf = bytearray(needed)
    received = 0
    while received < needed:
        view = memoryview(buf)[received:]
        chunk = sock.recv_into(view, needed - received)
        if chunk == 0:
            raise ConnectionError(
                f"socket closed mid-frame ({received}/{needed} bytes)"
            )
        received += chunk
    return list(struct.unpack(f"<{count}I", bytes(buf)))


def _write_uint32s(sock: socket.socket, values: list[int]) -> None:
    sock.sendall(struct.pack(f"<{len(values)}I", *values))


def _make_transport(
    num_draft_tokens: int = 4,
) -> tuple[RemoteTransport, socket.socket]:
    """Build a :class:`RemoteTransport` paired with a drafter-side socket."""
    target_sock, drafter_sock = _socket_pair()
    transport = RemoteTransport(num_draft_tokens=num_draft_tokens, sock=target_sock)
    return transport, drafter_sock


# ---------------------------------------------------------------------------
# RemoteTransport (target side) over a real socket pair
# ---------------------------------------------------------------------------


def test_open_session_allocates_unique_session_ids() -> None:
    transport, drafter_side = _make_transport()
    try:
        a = transport.open_session()
        b = transport.open_session()
        c = transport.open_session()
        assert a.session_id != b.session_id
        assert b.session_id != c.session_id
        assert a.num_draft_tokens == transport.num_draft_tokens
    finally:
        # Drain anything pending (forwarding ends + transport shutdown).
        # We never sent any commands, so the wire is clean. Close the
        # drafter side first so the transport's shutdown gets a clean
        # peer-closed signal instead of hanging on the ack recv.
        drafter_side.close()
        transport.shutdown()


def test_session_handle_forward_serialises_command_with_session_id() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()
        future = session.forward([42], num_forwards=4)

        cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        op, inputs, num_forwards, trim, sid, target_buf = _decode_command_frame(cmd)
        assert op == OP_FORWARD
        assert inputs == [42]
        assert num_forwards == 4
        assert trim == 0
        assert sid == session.session_id
        assert target_buf == session.num_draft_tokens + 1

        # Reply with K+1 = 5 drafts; the spec loop slices to num_forwards.
        _write_uint32s(drafter_side, [10, 11, 12, 13, 0])
        assert future.result() == [10, 11, 12, 13]
    finally:
        drafter_side.close()
        transport.shutdown()


def test_session_handle_trim_cache_emits_session_scoped_command() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()

        def _trim() -> None:
            session.trim_cache(3)

        thread = threading.Thread(target=_trim)
        thread.start()
        cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        op, _, _, trim, sid, _ = _decode_command_frame(cmd)
        assert op == OP_TRIM_CACHE
        assert trim == 3
        assert sid == session.session_id
        _write_uint32s(drafter_side, [ACK_OK])
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        drafter_side.close()
        transport.shutdown()


def test_session_handle_trim_cache_zero_is_noop() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()
        session.trim_cache(0)
        # Nothing must have been written: drafter_side.recv with
        # MSG_DONTWAIT should fail with BlockingIOError.
        drafter_side.setblocking(False)
        with pytest.raises(BlockingIOError):
            drafter_side.recv(1)
    finally:
        drafter_side.setblocking(True)
        drafter_side.close()
        transport.shutdown()


def test_session_handle_reset_and_prefill_sends_command_array_and_recv_ack() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()
        prompt = [101, 102, 103, 104, 105]

        def _prefill() -> None:
            session.reset_and_prefill(prompt)

        thread = threading.Thread(target=_prefill)
        thread.start()

        cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        op, inputs, num_forwards, trim, sid, _ = _decode_command_frame(cmd)
        assert op == OP_PREFILL
        assert inputs == []
        assert num_forwards == len(prompt)
        assert trim == 0
        assert sid == session.session_id

        # Length-prefixed prompt tail: 1 uint32 header + N tokens.
        header = _read_uint32s(drafter_side, 1)[0]
        assert header == len(prompt)
        tokens = _read_uint32s(drafter_side, len(prompt))
        assert tokens == prompt

        _write_uint32s(drafter_side, [ACK_OK])
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        drafter_side.close()
        transport.shutdown()


def test_session_handle_reset_and_prefill_empty_prompt_skips_array_send() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()

        def _prefill() -> None:
            session.reset_and_prefill([])

        thread = threading.Thread(target=_prefill)
        thread.start()

        cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        op, _, num_forwards, _, _, _ = _decode_command_frame(cmd)
        assert op == OP_PREFILL
        assert num_forwards == 0

        # No length-prefixed payload should follow on an empty prompt.
        # Confirm by acking immediately and joining.
        _write_uint32s(drafter_side, [ACK_OK])
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        drafter_side.close()
        transport.shutdown()


def test_session_handle_shutdown_sends_op_end_session() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()

        def _shutdown() -> None:
            session.shutdown()

        thread = threading.Thread(target=_shutdown)
        thread.start()

        cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        op, _, _, _, sid, _ = _decode_command_frame(cmd)
        assert op == OP_END_SESSION
        assert sid == session.session_id
        _write_uint32s(drafter_side, [ACK_OK])
        thread.join(timeout=2.0)
        assert not thread.is_alive()

        # Idempotent: a second shutdown is a no-op (no new wire op).
        session.shutdown()
        drafter_side.setblocking(False)
        with pytest.raises(BlockingIOError):
            drafter_side.recv(1)
    finally:
        drafter_side.setblocking(True)
        drafter_side.close()
        transport.shutdown()


def test_session_handle_rejects_use_after_shutdown() -> None:
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()

        def _shutdown_then_ack() -> None:
            cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
            op, _, _, _, _, _ = _decode_command_frame(cmd)
            assert op == OP_END_SESSION
            _write_uint32s(drafter_side, [ACK_OK])

        thread = threading.Thread(target=_shutdown_then_ack)
        thread.start()
        session.shutdown()
        thread.join(timeout=2.0)

        with pytest.raises(RuntimeError, match="after shutdown"):
            _ = session.forward([1], num_forwards=2)
        with pytest.raises(RuntimeError, match="after shutdown"):
            session.trim_cache(2)
        with pytest.raises(RuntimeError, match="after shutdown"):
            session.reset_and_prefill([1, 2, 3])
    finally:
        drafter_side.close()
        transport.shutdown()


def test_remote_transport_shutdown_sends_op_and_drains_executor() -> None:
    transport, drafter_side = _make_transport()
    try:

        def _ack_shutdown() -> None:
            cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
            op, _, _, _, _, _ = _decode_command_frame(cmd)
            assert op == OP_SHUTDOWN
            _write_uint32s(drafter_side, [ACK_OK])

        thread = threading.Thread(target=_ack_shutdown)
        thread.start()
        transport.shutdown()
        thread.join(timeout=2.0)

        # Idempotent: a second shutdown is a no-op (no new wire op).
        transport.shutdown()
    finally:
        drafter_side.close()


def test_remote_transport_rejects_use_after_shutdown() -> None:
    transport, drafter_side = _make_transport()

    def _ack_shutdown() -> None:
        try:
            cmd = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
            op, _, _, _, _, _ = _decode_command_frame(cmd)
            assert op == OP_SHUTDOWN
            _write_uint32s(drafter_side, [ACK_OK])
        except (ConnectionError, OSError):
            pass

    thread = threading.Thread(target=_ack_shutdown)
    thread.start()
    transport.shutdown()
    thread.join(timeout=2.0)

    with pytest.raises(RuntimeError, match="after shutdown"):
        _ = transport.open_session()
    drafter_side.close()


def test_remote_transport_rejects_invalid_num_draft_tokens() -> None:
    target_sock, drafter_sock = _socket_pair()
    try:
        with pytest.raises(ValueError, match="num_draft_tokens"):
            RemoteTransport(num_draft_tokens=0, sock=target_sock)
    finally:
        target_sock.close()
        drafter_sock.close()


# ---------------------------------------------------------------------------
# Drafter-death recovery: ``RemoteTransport.is_failed`` flag
# ---------------------------------------------------------------------------


def test_remote_transport_is_failed_starts_false() -> None:
    """A freshly-constructed transport is healthy."""
    transport, drafter_side = _make_transport()
    try:
        assert transport.is_failed is False
    finally:
        drafter_side.close()
        transport.shutdown()


def test_remote_transport_marks_failed_when_drafter_closes_mid_forward() -> None:
    """The blocking forward helper flips ``is_failed`` on socket close.

    Pre-fix failure mode: a peer-side close mid-frame raised
    ``ConnectionError`` once but left the transport looking healthy,
    so subsequent ``open_session`` calls would happily allocate a
    fresh session against a dead wire and the spec loop would re-
    discover the failure on every request.
    """
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()
        future = session.forward([42], num_forwards=4)
        # Drain the command frame so the drafter side is in a known
        # state, then close it before responding -- this models a
        # drafter rank that crashed after receiving the request.
        _ = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        drafter_side.close()
        with pytest.raises((ConnectionError, OSError)):
            future.result(timeout=2.0)
        assert transport.is_failed is True
    finally:
        # ``shutdown`` is best-effort against a dead wire; the
        # contextlib.suppress inside it swallows the secondary error.
        transport.shutdown()


def test_remote_transport_open_session_rejects_after_failure() -> None:
    """Once a wire-level failure has surfaced, no new session is allowed.

    Subsequent requests must NOT allocate a fresh session on a known-
    dead wire -- the runner will be torn down by the master's
    instance-deletion path and a new placement issued. ``open_session``
    raising RuntimeError is the fail-loud signal that bridges that
    gap.
    """
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()
        future = session.forward([42], num_forwards=4)
        _ = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        drafter_side.close()
        with pytest.raises((ConnectionError, OSError)):
            future.result(timeout=2.0)
        assert transport.is_failed is True
        with pytest.raises(RuntimeError, match="wire-level failure"):
            _ = transport.open_session()
    finally:
        transport.shutdown()


def test_remote_transport_marks_failed_when_drafter_closes_mid_trim() -> None:
    """The trim helper also flips ``is_failed`` on socket close.

    Trim is on the cache-reconciliation path between rounds; failure
    here surfaces the same way as a forward failure and must mark
    the transport so the next request fails fast.
    """
    transport, drafter_side = _make_transport()
    try:
        session = transport.open_session()

        def _do_trim() -> Exception | None:
            try:
                session.trim_cache(3)
            except Exception as exc:
                return exc
            return None

        result_box: list[Exception | None] = []

        def _runner() -> None:
            result_box.append(_do_trim())

        thread = threading.Thread(target=_runner)
        thread.start()
        # Drain the command frame, then drop the connection without
        # acking -- mid-trim drafter death.
        _ = _read_uint32s(drafter_side, COMMAND_FRAME_SIZE)
        drafter_side.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert isinstance(result_box[0], (ConnectionError, OSError))
        assert transport.is_failed is True
    finally:
        transport.shutdown()


# ---------------------------------------------------------------------------
# drafter_serve_loop dispatch
# ---------------------------------------------------------------------------


def _empty_cache_factory() -> object:
    """Drop-in factory for tests that don't actually run forwards."""
    return []


def test_drafter_serve_loop_handles_shutdown_immediately() -> None:
    """A bare OP_SHUTDOWN frame must terminate the serve loop with an ACK."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    target_sock, drafter_sock = _socket_pair()
    try:
        # Write the shutdown frame from the target side BEFORE entering
        # the serve loop so the recv inside the loop completes
        # immediately.
        shutdown_frame = _build_command_frame(
            op=OP_SHUTDOWN,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=SESSION_ID_NONE,
            target_drafts_buffer_size=5,
        )
        _write_uint32s(target_sock, shutdown_frame)

        drafter_serve_loop(
            draft_model=None,  # pyright: ignore[reportArgumentType]
            make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
            num_draft_tokens=4,
            sock=drafter_sock,
        )

        ack = _read_uint32s(target_sock, ACK_FRAME_SIZE)
        assert ack[0] == ACK_OK
    finally:
        target_sock.close()
        drafter_sock.close()


def test_drafter_serve_loop_handles_end_session_for_unknown_session() -> None:
    """``OP_END_SESSION`` for an unknown session is a successful no-op ack.

    Idempotent semantics: a target that crashed without sending
    ``OP_END_SESSION`` for a session is cleaned up by the next
    ``OP_SHUTDOWN``; targets that retry ``OP_END_SESSION`` after a
    transient network error still see an ack.
    """
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    target_sock, drafter_sock = _socket_pair()
    try:
        end_frame = _build_command_frame(
            op=OP_END_SESSION,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=99,
            target_drafts_buffer_size=5,
        )
        shutdown_frame = _build_command_frame(
            op=OP_SHUTDOWN,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=SESSION_ID_NONE,
            target_drafts_buffer_size=5,
        )
        _write_uint32s(target_sock, end_frame)
        _write_uint32s(target_sock, shutdown_frame)

        drafter_serve_loop(
            draft_model=None,  # pyright: ignore[reportArgumentType]
            make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
            num_draft_tokens=4,
            sock=drafter_sock,
        )

        ack_end = _read_uint32s(target_sock, ACK_FRAME_SIZE)
        ack_shutdown = _read_uint32s(target_sock, ACK_FRAME_SIZE)
        assert ack_end[0] == ACK_OK
        assert ack_shutdown[0] == ACK_OK
    finally:
        target_sock.close()
        drafter_sock.close()


def test_drafter_serve_loop_rejects_unknown_op() -> None:
    """An unknown op code must crash the serve loop loudly."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    target_sock, drafter_sock = _socket_pair()
    try:
        # Hand-build an unknown op code (255 is not a defined op).
        bogus = [255, 0, 0, 0, 0, 0, 0, 0, 0]
        _write_uint32s(target_sock, bogus)

        with pytest.raises(RuntimeError, match="Unknown op code"):
            drafter_serve_loop(
                draft_model=None,  # pyright: ignore[reportArgumentType]
                make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
                num_draft_tokens=4,
                sock=drafter_sock,
            )
    finally:
        target_sock.close()
        drafter_sock.close()


def test_drafter_serve_loop_rejects_op_for_unknown_session() -> None:
    """``OP_TRIM_CACHE`` / ``OP_FORWARD`` against an unallocated session crashes."""
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    target_sock, drafter_sock = _socket_pair()
    try:
        trim_frame = _build_command_frame(
            op=OP_TRIM_CACHE,
            inputs=[],
            num_forwards=0,
            trim_amount=2,
            session_id=42,
            target_drafts_buffer_size=5,
        )
        _write_uint32s(target_sock, trim_frame)

        with pytest.raises(RuntimeError, match="OP_TRIM_CACHE for unknown session"):
            drafter_serve_loop(
                draft_model=None,  # pyright: ignore[reportArgumentType]
                make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
                num_draft_tokens=4,
                sock=drafter_sock,
            )
    finally:
        target_sock.close()
        drafter_sock.close()


def test_drafter_serve_loop_rejects_reverse_k_drift() -> None:
    """``OP_FORWARD`` with mismatched buffer sizes (drafter K > target K) crashes.

    Regression test for the reverse-drift wire desync: when the
    drafter is configured with ``num_draft_tokens=4`` (buffer 5) but
    the target is configured with ``num_draft_tokens=2`` (buffer 3),
    the drafter would otherwise pad replies to 5 uint32s while the
    target reads only 3, leaving 2 ints in the socket buffer to be
    misinterpreted as part of the next command frame. The symmetric
    drift guard catches this BEFORE the response is sent.
    """
    from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

    target_sock, drafter_sock = _socket_pair()
    try:
        # Target advertises K+1 = 3 (num_draft_tokens=2); drafter
        # below is started with num_draft_tokens=4 (K+1 = 5).
        forward_frame = _build_command_frame(
            op=OP_FORWARD,
            inputs=[1],
            num_forwards=2,
            trim_amount=0,
            session_id=0,
            target_drafts_buffer_size=3,
        )
        _write_uint32s(target_sock, forward_frame)

        with pytest.raises(RuntimeError, match=r"OP_FORWARD wire-size mismatch"):
            drafter_serve_loop(
                draft_model=None,  # pyright: ignore[reportArgumentType]
                make_draft_cache=_empty_cache_factory,  # pyright: ignore[reportArgumentType]
                num_draft_tokens=4,
                sock=drafter_sock,
            )
    finally:
        target_sock.close()
        drafter_sock.close()


# Used by other tests that need to import _ from this module without
# triggering "unused" linter errors on intermediate Iterator hints.
_ = Iterator
