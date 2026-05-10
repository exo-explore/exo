"""Drafter on a different node, IPC via a direct TCP socket.

:class:`RemoteTransport` (a concrete :class:`DrafterTransport`) and the
matching :func:`drafter_serve_loop` carry the same uint32-array wire
protocol that the original implementation rode on top of
``mx.distributed.send/recv``, but they no longer require the drafter to
be a member of any ``mx.distributed.Group``.

Why the change: ``mx.distributed`` on Apple Silicon (jaccl, ring) does
not implement ``Group.split``. As long as the drafter rank shared the
parent group with the target ranks, the target ranks could not run
TP/PP collectives without dragging the drafter in -- the V1 asymmetric
path was therefore limited to a single target rank. By moving the
drafter wire onto a plain TCP socket, the parent ``mx.distributed``
group contains only target ranks (so target collectives work as
designed), and the drafter rank skips ``mx.distributed.init`` entirely.
The same code path works for parent_size 1 (single target) and
parent_size N (sharded target) without any backend feature gate.

Wire protocol v3 (session-aware, socket-framed -- semantically
identical to v2 but without the ``mx.distributed`` framing):

  * **Command frame** (target -> drafter), :data:`COMMAND_FRAME_SIZE`
    little-endian uint32s::

        [op, num_inputs, num_forwards, input_0, input_1, trim_amount,
         session_id, _, _]

    Fixed length so the receiver can call
    :func:`drafter_socket.recv_uint32_frame` with a known shape.
    ``session_id`` selects which per-session draft cache the drafter
    rank routes the op to. ``OP_SHUTDOWN`` ignores ``session_id``
    (it tears down the entire serve loop). All other ops require a
    valid ``session_id`` -- :data:`OP_PREFILL` allocates the session,
    :data:`OP_END_SESSION` frees it, the rest reference an existing
    session. Unused slots are zero-padded.

  * **Drafts frame** (drafter -> target), :data:`COMMAND_FRAME_SIZE` -
    sized? No: the drafts buffer is sized to ``num_draft_tokens + 1``.
    The target knows the buffer width statically from
    :attr:`RemoteTransport.num_draft_tokens`. Padded with zeros if the
    request asked for fewer than ``K + 1`` forwards (the caller knows
    its requested count and slices accordingly).

  * **Ack frame** (drafter -> target), :data:`ACK_FRAME_SIZE` uint32s:
    a single status byte (always ``0`` for "ok"). Sent after
    ``OP_TRIM_CACHE``, ``OP_PREFILL``, ``OP_END_SESSION``, and
    ``OP_SHUTDOWN`` so the target rank has a synchronisation point
    against the drafter's cache state.

  * **OP_PREFILL prompt tail** (target -> drafter): when the command
    frame's ``num_forwards`` slot is non-zero, the target follows the
    command frame with a length-prefixed prompt-token payload (see
    :func:`drafter_socket.send_variable_uint32_payload`). Empty
    prompts skip the tail entirely.

Op codes: :data:`OP_FORWARD` (1), :data:`OP_TRIM_CACHE` (2),
:data:`OP_SHUTDOWN` (3), :data:`OP_PREFILL` (4),
:data:`OP_END_SESSION` (5).

Concurrency model: ``RemoteTransport`` exposes :meth:`open_session`
which allocates a fresh ``session_id`` and returns a session-scoped
:class:`DrafterTransport` view. Each in-flight target request gets
its own session handle; the underlying wire stays serial because a
single TCP connection cannot interleave reads/writes from multiple
threads, but the drafter rank multiplexes operations across sessions
by keying each op's KV-cache lookup on ``session_id``. The cap on
concurrent target requests is therefore set by the *target* runner
(``EXO_MAX_CONCURRENT_REQUESTS``), not by the drafter wire.

Topology assumption: target rank 0 binds a TCP listener at instance
bootstrap; the drafter dials it. Address discovery flows through
:class:`DrafterPlacement` (host = target rank 0's advertised address,
port = ephemeral port allocated at placement time). One TCP
connection per asymmetric instance is sufficient because ops serialise
on a single socket.
"""

from __future__ import annotations

import contextlib
import itertools
import socket
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Final, final

from exo.worker.engines.mlx.generator.drafter_socket import (
    recv_uint32_frame,
    send_uint32_frame,
    send_variable_uint32_payload,
)

if TYPE_CHECKING:
    from exo.worker.engines.mlx.generator.drafter_transport import DraftFuture
    from exo.worker.engines.mlx.types import KVCacheType, Model

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache

# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

COMMAND_FRAME_SIZE: Final[int] = 9
"""Fixed size of a command frame (uint32 ints).

Carries [op, num_inputs, num_forwards, input_0, input_1, trim_amount,
session_id, target_drafts_buffer_size, 0]. The trailing zero slot is
reserved for future extension without bumping the wire version on the
byte layer."""

ACK_FRAME_SIZE: Final[int] = 1
"""Fixed size of an ack frame (uint32 ints). The single int is reserved
for a status code; ``0`` means ok. Future revisions may surface error
states here without changing the wire format."""

OP_FORWARD: Final[int] = 1
"""Drafter runs ``num_forwards`` forwards starting from
``inputs[:num_inputs]`` against ``sessions[session_id]``'s KV cache.
Replies with a Drafts frame."""

OP_TRIM_CACHE: Final[int] = 2
"""Drafter trims ``trim_amount`` positions from
``sessions[session_id]``'s KV cache. Replies with an Ack frame so the
target has a sync point."""

OP_SHUTDOWN: Final[int] = 3
"""Drafter exits its serve loop. Replies with an Ack frame, then the
serve loop returns. ``session_id`` is ignored -- this op tears down
the entire wire, not a single session. Per-session cleanup uses
:data:`OP_END_SESSION` instead."""

OP_PREFILL: Final[int] = 4
"""Per-request setup: target announces a prompt of ``num_inputs`` (used
as ``num_prompt_tokens``) tokens for ``session_id``. The drafter
allocates a fresh KV cache for the session (or resets the existing
one to offset 0), recvs the prompt token array, runs prefill forwards
through the drafter model, then replies with an Ack frame. Issued
once at the start of every request so the spec loop's first
``OP_FORWARD`` seeds against an aligned drafter cache."""

OP_END_SESSION: Final[int] = 5
"""Per-request teardown: drafter drops ``sessions[session_id]`` to free
the KV cache memory and replies with an Ack frame so the target has a
sync point. Idempotent: ending a non-existent session is also a
successful ack (sessions can drop themselves on the drafter side via
target shutdown without the target getting a chance to send this op).
"""

ACK_OK: Final[int] = 0

SESSION_ID_NONE: Final[int] = 0xFFFFFFFF
"""Sentinel ``session_id`` for ops that don't address a session.

``OP_SHUTDOWN`` carries this value because it tears down the whole
wire, not a single session. ``0`` is the first session id allocated by
the target's monotonic counter, so a sentinel out of that range avoids
a collision in wire-trace logs."""


def _build_command_frame(
    *,
    op: int,
    inputs: list[int],
    num_forwards: int,
    trim_amount: int,
    session_id: int,
    target_drafts_buffer_size: int,
) -> list[int]:
    """Pack command parameters into a fixed-length uint32 list.

    Layout: ``[op, num_inputs, num_forwards, input_0, input_1, trim_amount,
    session_id, target_drafts_buffer_size, 0]``.

    ``inputs`` must have length 0, 1, or 2 (the spec loop only ever
    passes length-1 or length-2 inputs to ``forward``; ``OP_TRIM_CACHE``,
    ``OP_END_SESSION``, and ``OP_SHUTDOWN`` pass length 0). Out-of-band
    lengths are a programming error and raise.

    ``session_id`` MUST fit in uint32. The target allocates session ids
    monotonically per :class:`RemoteTransport` instance from a counter,
    which gives ~4G sessions per runner lifetime -- plenty for any
    realistic deployment. Wraparound is not handled (the runner would
    have to serve > 4 billion concurrent requests; if that ever
    happens, switch the counter to a free-list of recycled ids).

    ``target_drafts_buffer_size`` is the target rank's local
    ``num_draft_tokens + 1``. The drafter validates it against its own
    ``drafts_buffer_size`` on ``OP_FORWARD`` so a mismatch -- the
    "drafter K > target K" reverse-drift case that the
    ``num_forwards > drafts_buffer_size`` guard cannot catch on its
    own -- fails fast with a clear runtime error instead of silently
    desyncing the wire (drafter would otherwise pad replies to its own
    ``drafts_buffer_size`` while the target reads only
    ``target_drafts_buffer_size``, leaving surplus bytes in the socket
    buffer that corrupt the next command frame). Carried on every
    frame so the drafter doesn't have to maintain per-session size
    state; one extra uint32 per command is negligible vs the prompt-
    tail payload.
    """
    if len(inputs) > 2:
        raise ValueError(f"inputs length must be in [0, 2], got {len(inputs)}")
    if not 0 <= session_id <= 0xFFFFFFFF:
        raise ValueError(f"session_id must fit in uint32, got {session_id}")
    if not 0 <= target_drafts_buffer_size <= 0xFFFFFFFF:
        raise ValueError(
            f"target_drafts_buffer_size must fit in uint32, "
            f"got {target_drafts_buffer_size}"
        )
    return [
        op,
        len(inputs),
        num_forwards,
        inputs[0] if len(inputs) >= 1 else 0,
        inputs[1] if len(inputs) >= 2 else 0,
        trim_amount,
        session_id,
        target_drafts_buffer_size,
        0,
    ]


def _decode_command_frame(
    flat: list[int],
) -> tuple[int, list[int], int, int, int, int]:
    """Inverse of :func:`_build_command_frame`.

    Returns ``(op, inputs, num_forwards, trim_amount, session_id,
    target_drafts_buffer_size)``.
    """
    if len(flat) != COMMAND_FRAME_SIZE:
        raise ValueError(
            f"Command frame has {len(flat)} ints, expected {COMMAND_FRAME_SIZE}"
        )
    op = flat[0]
    num_inputs = flat[1]
    num_forwards = flat[2]
    trim_amount = flat[5]
    session_id = flat[6]
    target_drafts_buffer_size = flat[7]
    inputs = flat[3 : 3 + num_inputs]
    return (
        op,
        inputs,
        num_forwards,
        trim_amount,
        session_id,
        target_drafts_buffer_size,
    )


# ---------------------------------------------------------------------------
# RemoteTransport (target side)
# ---------------------------------------------------------------------------


@final
class RemoteTransport:
    """Wire-protocol owner for the asymmetric drafter rank (target side).

    Holds the long-lived TCP socket + IPC thread; vends per-request
    :class:`_SessionHandle` instances via :meth:`open_session`. Each
    handle implements :class:`DrafterTransport` so the spec loop code
    is unchanged -- it just receives a session-scoped transport rather
    than the shared one.

    Each wire op (forward / trim / prefill / end-session) is dispatched
    on a single-worker :class:`ThreadPoolExecutor`. Wire ops therefore
    serialise even when multiple in-flight target requests are calling
    methods concurrently from different :class:`_SessionHandle`
    instances, which is exactly what we need: a single TCP connection
    cannot interleave reads/writes from multiple threads, but the
    drafter rank multiplexes operations across sessions by keying its
    KV-cache lookup on ``session_id``.

    Why a thread, given MLX is single-GIL? ``socket.recv`` blocks on
    the network until the peer responds; running the wire round-trip
    on a background thread lets the target's main thread issue MLX
    target-verify dispatches in parallel. The drafter's actual compute
    happens on the *drafter rank's* GPU, not on a thread of the calling
    rank, so there's no GIL contention to worry about.
    """

    def __init__(
        self,
        *,
        num_draft_tokens: int,
        sock: socket.socket,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        self._num_draft_tokens = num_draft_tokens
        self._sock = sock
        # Single-worker pool: every wire op (across all sessions) goes
        # through it serially, which keeps ``socket.send/recv`` safe
        # even when multiple :class:`_SessionHandle` instances are
        # in flight on different target tasks.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="exo-drafter-ipc"
        )
        self._is_shutdown = False
        # Sticky failure flag, set by the blocking wire helpers when a
        # socket-level error escapes (drafter rank crashed, peer closed
        # mid-frame, etc.). Once true, subsequent requests must not start
        # a new spec session on this transport: the wire is unrecoverable
        # until the runner is restarted (the master's instance-deletion
        # path tears the placement down within ~5s of the drafter node
        # leaving the topology). Callers consult :attr:`is_failed` before
        # constructing a :class:`PipelinedModelDrafter`; the runner
        # subprocess exits via the spec-loop exception if the failure
        # happens mid-request (see ``_pipelined_speculative_step``'s
        # abort sentinel).
        self._is_failed = False
        # Monotonic session id allocator. ``itertools.count`` gives us a
        # thread-safe unsigned counter; we wrap it in a lock-free
        # ``next()`` call inside :meth:`open_session` (Python's GIL
        # makes the increment atomic for CPython, but the lock makes
        # the contract explicit and survives a free-threaded build).
        self._session_id_counter = itertools.count()
        self._session_lock = threading.Lock()

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    @property
    def is_failed(self) -> bool:
        """True once a wire-level failure has been observed on this transport.

        Set by the blocking wire helpers when an :class:`OSError` escapes
        (drafter rank crashed, peer closed mid-frame, etc.). Sticky:
        there is no in-place recovery -- the runner must be torn down
        (via the master's instance-deletion path) and a fresh transport
        built. Callers consult this flag before constructing a
        :class:`PipelinedModelDrafter`; if true the request degrades to
        non-speculative decoding for the remaining lifetime of the
        runner.
        """
        return self._is_failed

    def _mark_failed(self) -> None:
        """Internal: flip :attr:`is_failed` to True. Idempotent."""
        self._is_failed = True

    def open_session(self) -> "_SessionHandle":
        """Allocate a fresh session and return a :class:`DrafterTransport` view.

        Each call yields a unique ``session_id``; the handle's
        :meth:`_SessionHandle.shutdown` sends ``OP_END_SESSION`` so the
        drafter rank can free the per-session KV cache. Forgetting to
        call :meth:`_SessionHandle.shutdown` leaks a KV cache on the
        drafter rank for that session id; ``RemoteTransport.shutdown``
        cleans up at process exit either way.
        """
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.open_session called after shutdown; the "
                "drafter rank's serve loop has exited and won't respond"
            )
        if self._is_failed:
            raise RuntimeError(
                "RemoteTransport.open_session called after a wire-level "
                "failure was observed; the underlying socket is dead and "
                "the runner must be torn down (master-driven instance "
                "deletion) before a fresh session can be opened"
            )
        with self._session_lock:
            session_id = next(self._session_id_counter)
        if session_id == SESSION_ID_NONE:
            # 4G sessions exhausted; bump again so we never collide
            # with the shutdown sentinel. In practice unreachable.
            with self._session_lock:
                session_id = next(self._session_id_counter)
        return _SessionHandle(owner=self, session_id=session_id)

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        # Send shutdown to the drafter and wait for the ack so the
        # drafter has a chance to drain its own state cleanly.
        try:
            self._executor.submit(self._shutdown_blocking).result(timeout=10.0)
        except Exception:
            # Drafter rank may already be torn down; the socket close
            # below cleans up regardless. The shutdown contract is
            # best-effort: if the wire is broken there is nothing to
            # ack.
            pass
        finally:
            self._executor.shutdown(wait=True)
            with contextlib.suppress(OSError):
                self._sock.close()

    # -- session-scoped wire ops (called by _SessionHandle) -------------

    def _submit_forward(
        self, session_id: int, inputs: list[int], num_forwards: int
    ) -> "DraftFuture":
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.forward called after shutdown; the drafter "
                "rank's serve loop has exited and won't respond"
            )
        upper = self._num_draft_tokens + 1
        if not 1 <= num_forwards <= upper:
            raise ValueError(
                f"num_forwards must be in [1, {upper}], got {num_forwards}"
            )
        if not 1 <= len(inputs) <= 2:
            raise ValueError(f"inputs must have length 1 or 2, got {len(inputs)}")
        return self._executor.submit(
            self._forward_blocking, session_id, inputs, num_forwards
        )

    def _submit_trim(self, session_id: int, n_positions: int) -> None:
        if self._is_shutdown:
            raise RuntimeError("RemoteTransport.trim_cache called after shutdown")
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions == 0:
            return
        self._executor.submit(self._trim_blocking, session_id, n_positions).result()

    def _submit_prefill(self, session_id: int, prompt_tokens: list[int]) -> None:
        if self._is_shutdown:
            raise RuntimeError(
                "RemoteTransport.reset_and_prefill called after shutdown"
            )
        self._executor.submit(
            self._reset_and_prefill_blocking, session_id, prompt_tokens
        ).result()

    def _submit_end_session(self, session_id: int) -> None:
        # Best-effort: if the wire is already shut down (process is
        # tearing down), the session-side OP_END_SESSION would fail
        # but the drafter rank is also exiting, so the cache is freed
        # by process death anyway.
        if self._is_shutdown:
            return
        self._executor.submit(self._end_session_blocking, session_id).result()

    # -- internals --------------------------------------------------------

    def _forward_blocking(
        self, session_id: int, inputs: list[int], num_forwards: int
    ) -> list[int]:
        """Send a forward command and recv the drafts. Runs on the IPC thread."""
        frame = _build_command_frame(
            op=OP_FORWARD,
            inputs=inputs,
            num_forwards=num_forwards,
            trim_amount=0,
            session_id=session_id,
            target_drafts_buffer_size=self._num_draft_tokens + 1,
        )
        try:
            send_uint32_frame(self._sock, frame)
            # Drafts buffer is fixed-size at K + 1 (the upper bound of any
            # forward request); we slice to ``num_forwards`` here.
            drafts = recv_uint32_frame(self._sock, self._num_draft_tokens + 1)
        except OSError:
            # Drafter rank closed the socket / peer reset / broken pipe.
            # Mark the transport so subsequent ``open_session`` calls
            # fail fast and the runner can be torn down (master-driven
            # instance deletion) instead of silently producing nothing
            # on every speculative round.
            self._mark_failed()
            raise
        return drafts[:num_forwards]

    def _trim_blocking(self, session_id: int, n_positions: int) -> None:
        """Send a trim command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_TRIM_CACHE,
            inputs=[],
            num_forwards=0,
            trim_amount=n_positions,
            session_id=session_id,
            target_drafts_buffer_size=self._num_draft_tokens + 1,
        )
        try:
            send_uint32_frame(self._sock, frame)
            ack = recv_uint32_frame(self._sock, ACK_FRAME_SIZE)
        except OSError:
            self._mark_failed()
            raise
        if ack[0] != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {ack[0]} "
                f"for trim_cache(session={session_id}, n={n_positions})"
            )

    def _shutdown_blocking(self) -> None:
        """Send shutdown command and wait for the ack."""
        frame = _build_command_frame(
            op=OP_SHUTDOWN,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=SESSION_ID_NONE,
            target_drafts_buffer_size=self._num_draft_tokens + 1,
        )
        send_uint32_frame(self._sock, frame)
        # Best-effort recv: if the drafter has already torn down, the
        # peer close will surface here. The caller is shutting down
        # either way, so swallow recv failures.
        with contextlib.suppress(ConnectionError, OSError):
            recv_uint32_frame(self._sock, ACK_FRAME_SIZE)

    def _reset_and_prefill_blocking(
        self, session_id: int, prompt_tokens: list[int]
    ) -> None:
        """Send the prefill command + token array and wait for the ack.

        The command frame announces ``num_prompt_tokens`` (encoded in
        the ``num_forwards`` slot) and the ``session_id`` to allocate /
        reset on the drafter rank. The prompt tail follows immediately
        when non-empty, length-prefixed for parser robustness.
        """
        num_prompt_tokens = len(prompt_tokens)
        frame = _build_command_frame(
            op=OP_PREFILL,
            inputs=[],
            num_forwards=num_prompt_tokens,
            trim_amount=0,
            session_id=session_id,
            target_drafts_buffer_size=self._num_draft_tokens + 1,
        )
        try:
            send_uint32_frame(self._sock, frame)
            if num_prompt_tokens > 0:
                send_variable_uint32_payload(self._sock, prompt_tokens)
            ack = recv_uint32_frame(self._sock, ACK_FRAME_SIZE)
        except OSError:
            self._mark_failed()
            raise
        if ack[0] != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {ack[0]} "
                f"for reset_and_prefill(session={session_id}, "
                f"{num_prompt_tokens} tokens)"
            )

    def _end_session_blocking(self, session_id: int) -> None:
        """Send OP_END_SESSION and wait for the ack."""
        frame = _build_command_frame(
            op=OP_END_SESSION,
            inputs=[],
            num_forwards=0,
            trim_amount=0,
            session_id=session_id,
            target_drafts_buffer_size=self._num_draft_tokens + 1,
        )
        try:
            send_uint32_frame(self._sock, frame)
            ack = recv_uint32_frame(self._sock, ACK_FRAME_SIZE)
        except OSError:
            self._mark_failed()
            raise
        if ack[0] != ACK_OK:
            raise RuntimeError(
                f"Drafter rank reported error code {ack[0]} "
                f"for end_session({session_id})"
            )


@final
class _SessionHandle:
    """Per-request :class:`DrafterTransport` view of a :class:`RemoteTransport`.

    Each in-flight target task gets its own handle via
    :meth:`RemoteTransport.open_session`. The handle's wire ops carry
    the handle's ``session_id`` so the drafter rank can route them to
    the right per-session KV cache.

    Lifecycle:

    * :meth:`reset_and_prefill` allocates the session on the drafter
      rank and seeds its KV cache with the prompt prefix.
    * :meth:`forward` / :meth:`trim_cache` advance / rollback the
      session's KV cache.
    * :meth:`shutdown` ends the session (sends ``OP_END_SESSION`` so
      the drafter rank frees the KV cache). Idempotent; safe to call
      from a generator's ``finally`` block.

    All methods raise :class:`RuntimeError` after :meth:`shutdown` so
    use-after-end mistakes surface immediately rather than corrupting
    a freshly allocated session that happens to reuse the id.
    """

    def __init__(self, *, owner: "RemoteTransport", session_id: int) -> None:
        self._owner = owner
        self._session_id = session_id
        self._closed = False

    @property
    def num_draft_tokens(self) -> int:
        return self._owner.num_draft_tokens

    @property
    def session_id(self) -> int:
        return self._session_id

    def forward(self, inputs: list[int], num_forwards: int) -> "DraftFuture":
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).forward called after shutdown"
            )
        return self._owner._submit_forward(self._session_id, inputs, num_forwards)  # pyright: ignore[reportPrivateUsage]

    def trim_cache(self, n_positions: int) -> None:
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).trim_cache called after shutdown"
            )
        self._owner._submit_trim(self._session_id, n_positions)  # pyright: ignore[reportPrivateUsage]

    def reset_and_prefill(self, prompt_tokens: list[int]) -> None:
        if self._closed:
            raise RuntimeError(
                f"_SessionHandle({self._session_id}).reset_and_prefill called after shutdown"
            )
        self._owner._submit_prefill(self._session_id, prompt_tokens)  # pyright: ignore[reportPrivateUsage]

    def shutdown(self) -> None:
        """End the session on the drafter rank. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._owner._submit_end_session(self._session_id)  # pyright: ignore[reportPrivateUsage]


def make_remote_transport(
    *,
    draft_model: "Model | None" = None,
    draft_cache: "KVCacheType | None" = None,
    num_draft_tokens: int,
    sock: socket.socket | None = None,
) -> "RemoteTransport":
    """Construct a :class:`RemoteTransport` for the calling target rank.

    Returns the wire-protocol owner; per-task callers should call
    :meth:`RemoteTransport.open_session` to obtain a session-scoped
    :class:`DrafterTransport` view that the spec loop consumes. The
    factory does not implement ``DrafterTransport`` directly because
    its lifecycle is bound to the runner (long-lived) while the spec
    loop's transport is bound to a single request (short-lived).

    Args:
        draft_model: Ignored (the model lives on the drafter rank).
            Included in the signature for parity with the in-process
            factory so callers don't branch on transport kind.
        draft_cache: Ignored (lives on the drafter rank).
        num_draft_tokens: ``K`` -- max drafts per round.
        sock: Connected TCP socket from target rank 0 to the drafter
            rank. The runner bootstrap accepts the drafter's incoming
            connection and hands the resulting socket here.

    Raises:
        ValueError: required kwargs missing.
    """
    del draft_model, draft_cache  # not relevant on target rank
    if sock is None:
        raise ValueError(
            "make_remote_transport requires `sock`; the asymmetric "
            "instance bootstrap accepts the drafter's incoming TCP "
            "connection and passes the connected socket here"
        )
    return RemoteTransport(
        num_draft_tokens=num_draft_tokens,
        sock=sock,
    )


# ---------------------------------------------------------------------------
# drafter_serve_loop (drafter side)
# ---------------------------------------------------------------------------


def drafter_serve_loop(
    *,
    draft_model: "Model",
    make_draft_cache: Callable[[], "KVCacheType"],
    num_draft_tokens: int,
    sock: socket.socket,
) -> None:
    """Run the drafter rank's command-loop until ``OP_SHUTDOWN``.

    Receives :data:`COMMAND_FRAME_SIZE`-element command frames over
    ``sock``, dispatches on the op code, executes the drafter-side
    work, and replies with the appropriate frame.

    Maintains a per-session KV cache (``sessions[session_id]``)
    allocated lazily on the first ``OP_PREFILL`` for each session and
    freed by ``OP_END_SESSION`` (or implicitly by ``OP_SHUTDOWN``).
    Multiple sessions may be live concurrently; the wire stays serial
    but the drafter rank multiplexes by ``session_id``.

    See module docstring for the wire protocol.
    """
    drafts_buffer_size = num_draft_tokens + 1
    sessions: dict[int, "KVCacheType"] = {}

    while True:
        flat = recv_uint32_frame(sock, COMMAND_FRAME_SIZE)
        (
            op,
            inputs,
            num_forwards,
            trim_amount,
            session_id,
            target_drafts_buffer_size,
        ) = _decode_command_frame(flat)

        if op == OP_SHUTDOWN:
            # Drop every session's cache before the serve loop returns
            # so the drafter rank's process exits with no dangling
            # KV-cache references holding GPU memory.
            sessions.clear()
            send_uint32_frame(sock, [ACK_OK])
            return

        if op == OP_END_SESSION:
            # Idempotent: ending a non-existent session is also a
            # successful ack. Forgetful targets (e.g. a runner that
            # crashed without calling shutdown on its session) are
            # cleaned up by the next ``OP_SHUTDOWN`` either way.
            sessions.pop(session_id, None)
            send_uint32_frame(sock, [ACK_OK])
            continue

        if op == OP_TRIM_CACHE:
            session_cache = sessions.get(session_id)
            if session_cache is None:
                raise RuntimeError(
                    f"OP_TRIM_CACHE for unknown session {session_id}; "
                    f"OP_PREFILL must allocate the session first"
                )
            if trim_amount > 0:
                # ``mlx_trim_prompt_cache`` is typed against ``List[Cache]``
                # but exo's ``KVCacheType`` is structurally a list of
                # mlx_lm caches; the runtime types match exactly. We
                # erase to ``Any`` here to bypass list invariance.
                from typing import Any
                from typing import cast as _cast

                mlx_trim_prompt_cache(_cast(Any, session_cache), trim_amount)  # type: ignore[reportArgumentType]
            send_uint32_frame(sock, [ACK_OK])
            continue

        if op == OP_FORWARD:
            # Wire-protocol invariants checked BEFORE any session
            # state lookup: the v3 reply is a fixed-width
            # ``drafts_buffer_size`` (== ``num_draft_tokens + 1``)
            # frame on every ``OP_FORWARD``. The target side calls
            # ``recv_uint32_frame(sock, target_drafts_buffer_size)``
            # and consumes exactly that many ints. The two sizes MUST
            # agree: any mismatch leaves bytes in the socket buffer
            # (drafter K > target K) or under-reads the response
            # (target K > drafter K), and either case corrupts the
            # next round-trip's command frame. Validating before the
            # session-cache lookup means a desynced wire fails with a
            # protocol-level error rather than an incidental
            # "unknown session" error caused by garbage in slot 6.
            #
            # Symmetric-drift guard, in priority order:
            #
            # 1. ``target_drafts_buffer_size != drafts_buffer_size`` --
            #    catches both directions of drift (target K > drafter
            #    K and drafter K > target K). Carried explicitly on
            #    the frame because the drafter cannot infer target's
            #    K from ``num_forwards`` alone (target may legitimately
            #    request fewer than its max under adaptive K).
            # 2. ``num_forwards > drafts_buffer_size`` -- defense in
            #    depth: implied by guard 1 in nominal cases, but a
            #    target that mistakenly sends ``num_forwards`` beyond
            #    its own buffer would otherwise tip ``_run_drafter_*``
            #    into an out-of-bounds slice.
            if target_drafts_buffer_size != drafts_buffer_size:
                raise RuntimeError(
                    f"OP_FORWARD wire-size mismatch: drafter "
                    f"drafts_buffer_size={drafts_buffer_size} "
                    f"(EXO_NUM_DRAFT_TOKENS={num_draft_tokens}), target "
                    f"target_drafts_buffer_size="
                    f"{target_drafts_buffer_size} (target K+1). Each "
                    f"side reads/writes its own size, so the surplus or "
                    f"shortfall would corrupt the next command frame. "
                    f"Restart the runner with the same "
                    f"EXO_NUM_DRAFT_TOKENS on every rank."
                )
            if num_forwards > drafts_buffer_size:
                raise RuntimeError(
                    f"OP_FORWARD num_forwards={num_forwards} exceeds "
                    f"wire-protocol budget drafts_buffer_size="
                    f"{drafts_buffer_size}; target requested more "
                    f"forwards than its own buffer can hold."
                )
            session_cache = sessions.get(session_id)
            if session_cache is None:
                raise RuntimeError(
                    f"OP_FORWARD for unknown session {session_id}; "
                    f"OP_PREFILL must allocate the session first"
                )
            outputs = _run_drafter_forwards_remote(
                draft_model=draft_model,
                draft_cache=session_cache,
                inputs=inputs,
                num_forwards=num_forwards,
            )
            # ``_run_drafter_forwards_remote`` is contracted to return
            # exactly ``num_forwards`` ints. The padding below is a
            # no-op when ``num_forwards == drafts_buffer_size`` and
            # zero-fills the trailing slots otherwise. Codex P1.5:
            # explicit assert + length comparison guards against an
            # ``outputs`` list longer than ``drafts_buffer_size`` that
            # would otherwise produce a *negative* multiplier on the
            # padding (silently truncating in surprising ways). Both
            # invariants are upheld by the ``num_forwards`` guard above
            # plus the contract of ``_run_drafter_forwards_remote``;
            # the asserts make the wire-protocol invariant explicit at
            # the point we compute the padded reply.
            assert len(outputs) == num_forwards, (
                f"drafter forwarded {len(outputs)} tokens, expected "
                f"{num_forwards} (wire-protocol invariant)"
            )
            assert len(outputs) <= drafts_buffer_size, (
                f"drafter outputs len={len(outputs)} exceeds "
                f"drafts_buffer_size={drafts_buffer_size}; the "
                f"num_forwards <= drafts_buffer_size guard above "
                f"should have prevented this"
            )
            padded = list(outputs) + [0] * (drafts_buffer_size - len(outputs))
            send_uint32_frame(sock, padded)
            continue

        if op == OP_PREFILL:
            # ``num_forwards`` is overloaded here as the prompt token
            # count (see _build_command_frame call site in
            # _reset_and_prefill_blocking).
            num_prompt_tokens = num_forwards
            # Allocate (or replace) the session's KV cache. Replacement
            # semantics let a target re-use a session_id after
            # OP_END_SESSION + OP_PREFILL without leaking the old cache.
            session_cache = make_draft_cache()
            sessions[session_id] = session_cache
            _reset_and_prefill_remote(
                draft_model=draft_model,
                draft_cache=session_cache,
                num_prompt_tokens=num_prompt_tokens,
                sock=sock,
            )
            send_uint32_frame(sock, [ACK_OK])
            continue

        # Unknown op code: this is a wire-protocol violation, not a
        # recoverable error. Raise so the serve loop dies and the
        # caller's ``RemoteTransport`` surfaces the broken-pipe error.
        raise RuntimeError(f"Unknown op code from target rank: {op}")


def _run_drafter_forwards_remote(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    inputs: list[int],
    num_forwards: int,
) -> list[int]:
    """Same forward semantics as ``InProcessTransport._run_drafter_forwards``.

    Kept as a free function to avoid importing the in-process transport
    on the drafter rank (which only loads the drafter model, not any
    target-side code).
    """
    if num_forwards < 1:
        raise ValueError(f"num_forwards must be >= 1, got {num_forwards}")
    if not 1 <= len(inputs) <= 2:
        raise ValueError(f"inputs must have length 1 or 2, got {len(inputs)}")
    ys: list[mx.array] = []
    y = mx.array(inputs, dtype=mx.uint32)
    for _ in range(num_forwards):
        logits = draft_model(y[None], cache=draft_cache)
        sampled = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.async_eval(sampled)
        ys.append(sampled)
        y = sampled
    mx.eval(ys + [c.state for c in draft_cache])  # type: ignore[reportArgumentType]
    return [int(t.item()) for t in ys]


_DRAFTER_PREFILL_STEP_SIZE: Final[int] = 4096
"""Chunk size for drafter-side prefill forwards.

Mirrors :func:`exo.worker.engines.mlx.generator.generate._spec_drafter_prefill`'s
``step`` default. Drafter weights are small (typically <2 GB) so the
4096-token chunks comfortably fit in the drafter rank's command queue
without OOM, even at long prompts."""


def _reset_and_prefill_remote(
    *,
    draft_model: "Model",
    draft_cache: "KVCacheType",
    num_prompt_tokens: int,
    sock: socket.socket,
) -> None:
    """Reset drafter cache and prefill against an incoming prompt.

    Pulled out as a free function (matches
    :func:`_run_drafter_forwards_remote`) so the drafter rank doesn't
    depend on any target-side code. The target rank already sent the
    ``OP_PREFILL`` command frame; this function handles the cache
    reset, recvs the prompt array (if any) over ``sock``, and runs the
    prefill forwards. The serve loop sends the ack after this returns.
    """
    # Trim cache to offset 0 so the new prompt starts cleanly. KVCache's
    # offset is the only state we need to reset; SSM caches and other
    # exotic types are not in scope for the drafter (drafter models are
    # standard transformers by convention). If the offset is 0 the trim
    # is a no-op.
    current_offset = 0
    if draft_cache:
        # Every cache entry shares the same offset for transformer
        # drafters; use entry 0 as the source of truth.
        cache_zero = draft_cache[0]
        offset_attr = getattr(cache_zero, "offset", None)
        if isinstance(offset_attr, int):
            current_offset = offset_attr
    if current_offset > 0:
        from typing import cast as _cast

        mlx_trim_prompt_cache(_cast(list[object], draft_cache), current_offset)  # type: ignore[reportArgumentType]

    if num_prompt_tokens == 0:
        return

    # Pull the prompt array from the target rank. The header preceding
    # the payload is sent by ``send_variable_uint32_payload`` and must
    # match ``num_prompt_tokens`` -- mismatches indicate a wire-protocol
    # bug rather than a recoverable error.
    header = recv_uint32_frame(sock, 1)
    received_count = header[0]
    if received_count != num_prompt_tokens:
        raise RuntimeError(
            f"OP_PREFILL prompt header mismatch: command announced "
            f"{num_prompt_tokens} tokens but payload header says "
            f"{received_count}"
        )
    prompt_tokens = recv_uint32_frame(sock, num_prompt_tokens)
    tokens = mx.array(prompt_tokens, dtype=mx.uint32)
    mx.eval(tokens)

    # Mirror :func:`_spec_drafter_prefill`: feed tokens through the
    # drafter model in chunks, advancing its KV cache.
    step = _DRAFTER_PREFILL_STEP_SIZE
    cursor = 0
    while cursor < num_prompt_tokens:
        chunk_end = min(cursor + step, num_prompt_tokens)
        chunk = tokens[cursor:chunk_end]
        draft_model(chunk[None], cache=draft_cache)
        mx.eval([c.state for c in draft_cache])  # type: ignore[reportArgumentType]
        cursor = chunk_end


__all__ = [
    "ACK_FRAME_SIZE",
    "ACK_OK",
    "COMMAND_FRAME_SIZE",
    "OP_END_SESSION",
    "OP_FORWARD",
    "OP_PREFILL",
    "OP_SHUTDOWN",
    "OP_TRIM_CACHE",
    "SESSION_ID_NONE",
    "RemoteTransport",
    "drafter_serve_loop",
    "make_remote_transport",
]


# Suppress the unused-import warnings for the future-only Future type:
# ThreadPoolExecutor.submit returns ``Future`` which is structurally
# compatible with :data:`DraftFuture`, but we annotate the return type
# inside the class body and the import is otherwise unused.
_ = Future
