"""Transport-agnostic interface to a speculative-decoding drafter.

The pipelined spec loop in :mod:`pipelined_drafter` orchestrates rounds
against a drafter through this Protocol, so the same loop drives an
in-process drafter (this module's :class:`InProcessTransport`) or a
drafter on a different MLX rank (:mod:`remote_drafter`'s
``RemoteTransport``, communicating via ``mx.distributed.send/recv`` over
the existing JACCL/ring backend -- RDMA over Thunderbolt-bridge between
twin Macs is just a backend choice the same call site honours).

API surface (kept as small as possible -- the spec loop owns
high-level accept/reject logic; the transport just implements the
mechanical primitives):

  * ``forward(inputs, num_forwards)`` -> ``Future[list[int]]`` -- run
    ``num_forwards`` drafter forwards. Forward 0 consumes ``inputs``
    (length 1 for partial-accept seeds, length 2 for full-accept
    seeds matching mlx_lm's ``draft_y = [drafts[-1], bonus]``
    convention); forwards 1..N-1 consume the previous forward's
    sampled output. Returns immediately so the caller can dispatch
    target verify in parallel.

    The spec loop uses this in two patterns:
      * Standard round: ``forward([seed], K)`` -> K drafts.
      * Speculative round (bonus prediction + round-ahead): ``forward([drafts[-1]], K+1)``
        -> ``[d_K, d^spec_0, ..., d^spec_{K-1}]`` where ``d_K`` is the
        drafter's prediction for the bonus position (compared against
        the actual ``bonus_t`` to detect speculation hit).
  * ``trim_cache(n)`` -- trim ``n`` positions from the drafter's KV
    cache. Used after partial accept (trim rejected drafts) and after
    speculation miss (rollback the speculative forward).
  * ``shutdown()`` -- release transport resources. No-op for the
    in-process transport.

The Future returned by ``propose`` is a synchronous
:class:`concurrent.futures.Future`, not :mod:`asyncio`. The spec loop
is a synchronous generator; blocking on a sync Future from a generator
is natural, whereas threading asyncio through the generator would be
invasive. The remote transport's IPC thread sets the Future from
outside the calling thread, which ``concurrent.futures.Future``
supports.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Callable, Final, Protocol, final, runtime_checkable

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache

from exo.worker.engines.mlx.types import KVCacheType, Model

# Returned by ``propose``; the spec loop blocks on ``.result()`` once it
# has dispatched target verify.
DraftFuture = Future[list[int]]


@runtime_checkable
class DrafterTransport(Protocol):
    """Async access to a speculative-decoding drafter.

    Implementations MUST be safe under the call sequence
    :func:`pipelined_speculative_step` issues:

      1. ``forward([seed], K)`` -> ``future``
      2. (caller dispatches target verify in parallel)
      3. ``future.result()``
      4. either:
         a. partial accept: ``trim_cache(K - num_accepted - 1)`` then
            ``forward([target_correction], K)`` for next round, or
         b. full accept: no trim, then ``forward([drafts[-1], bonus], K)``
            for next round.

    For cross-round speculation an additional ``forward([drafts[-1]], K+1)``
    is issued in step 2 (parallel with verify); the first of the K+1
    returned tokens is the drafter's predicted bonus, which is checked
    against the actual ``bonus_t``. On hit, the remaining K outputs are
    used as round t+1's drafts. On miss, ``trim_cache(K + 1)`` rolls
    back the speculative work.

    Behaviour is undefined if more than one un-resolved Future is in
    flight without an intervening ``trim_cache`` or ``.result()`` call.
    """

    @property
    def num_draft_tokens(self) -> int:
        """``K`` -- the typical number of drafts per round.

        Remote transports use this to pre-allocate fixed-size receive
        buffers (sized for ``K + 1`` to cover the speculative forward).
        ``forward()`` accepts ``num_forwards`` up to ``K + 1``.
        """
        ...

    def forward(self, inputs: list[int], num_forwards: int) -> DraftFuture:
        """Run ``num_forwards`` drafter forwards starting from ``inputs``.

        Args:
            inputs: First-forward input. Length 1 for partial-accept
                seeds (``[seed]``); length 2 for full-accept seeds
                (``[drafts[-1], bonus]`` matching mlx_lm's
                ``_draft_generate`` ``draft_y`` convention).
                Subsequent forwards consume the previous forward's
                output, so they are always length-1.
            num_forwards: Number of forwards (and number of returned
                sampled tokens). Must satisfy
                ``1 <= num_forwards <= self.num_draft_tokens + 1``;
                the ``+ 1`` covers the speculative bonus-prediction
                forward.

        Cache effect: extends the drafter's KV cache by
        ``len(inputs) + num_forwards - 1`` positions.

        Returns:
            A Future resolving to ``num_forwards`` sampled token ids.
        """
        ...

    def trim_cache(self, n_positions: int) -> None:
        """Trim ``n_positions`` from the drafter's KV cache.

        Used after partial accept (``n_positions = K - num_accepted - 1``)
        and after speculation miss (``n_positions = positions added by
        the speculative forward``).

        ``n_positions == 0`` is a valid no-op so callers don't have to
        guard against the trivial case. Negative values raise
        ``ValueError``.
        """
        ...

    def reset_and_prefill(self, prompt_tokens: list[int]) -> None:
        """Reset the drafter cache and prefill it with ``prompt_tokens``.

        Issued once at the start of every request so the drafter cache
        is aligned with the target's cache before the spec loop starts.
        ``prompt_tokens`` is the prompt minus the last 2 tokens (matching
        the in-process path's ``_spec_drafter_prefill`` invariant);
        the spec loop seeds from the last prompt token internally.

        Empty ``prompt_tokens`` is valid (very short prompts) and only
        resets the cache.

        For the in-process transport this is a no-op when the caller
        owns drafter cache prefill externally (the legacy mlx_generate
        path). Implementations that own the drafter cache fully (e.g.
        the remote transport) handle reset + prefill internally here.
        """
        ...

    def shutdown(self) -> None:
        """Release transport resources. Idempotent.

        In-process transport: no-op (the drafter model and cache are
        owned by the caller). Remote transport: terminates the drafter
        rank's serve loop, drains pending IPC.
        """
        ...


# ---------------------------------------------------------------------------
# In-process transport
# ---------------------------------------------------------------------------


@final
class InProcessTransport:
    """Drafter model + cache live in the calling process on the same MLX device.

    All MLX work happens on the calling thread; ``propose`` runs the K
    drafter forwards inline and returns an immediately-resolved Future
    so the call site is uniform with the remote transport. Any
    pipelining win at this transport comes from MLX's intra-forward
    async dispatch (``mx.async_eval`` between drafter forwards) and
    the cross-round speculation in :func:`pipelined_speculative_step`.

    Apple Silicon's unified-memory single GPU bounds the gain because
    drafter and target target compete for the same memory bandwidth on
    the same Metal command queue; on multi-machine deployments the
    same call site runs against :class:`RemoteTransport` instead and
    the gain unlocks.
    """

    def __init__(
        self,
        *,
        draft_model: Model,
        draft_cache: KVCacheType,
        num_draft_tokens: int,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        self._draft_model = draft_model
        self._draft_cache = draft_cache
        self._num_draft_tokens = num_draft_tokens

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def forward(self, inputs: list[int], num_forwards: int) -> DraftFuture:
        # ``+ 1`` upper bound covers the speculative bonus-prediction
        # forward; see DrafterTransport docstring.
        upper = self._num_draft_tokens + 1
        if not 1 <= num_forwards <= upper:
            raise ValueError(
                f"num_forwards must be in [1, {upper}], got {num_forwards}"
            )
        if not 1 <= len(inputs) <= 2:
            # Length 1 = partial-accept seed; length 2 = full-accept
            # ``[drafts[-1], bonus]`` shape. No other shape is meaningful
            # for spec decoding and accepting it would mask bookkeeping
            # bugs in the spec loop.
            raise ValueError(f"inputs must have length 1 or 2, got {len(inputs)}")

        future: DraftFuture = Future()
        try:
            outputs = self._run_drafter_forwards(inputs, num_forwards)
            future.set_result(outputs)
        except Exception as exc:
            future.set_exception(exc)
        return future

    def trim_cache(self, n_positions: int) -> None:
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions == 0:
            return
        # mlx_lm types ``trim_prompt_cache`` against ``List[Cache]``;
        # exo's ``KVCacheType`` is a structural superset, hence the
        # cast + ignore (same pattern used in ``mlx_generate`` and the
        # n-gram spec loop).
        from typing import cast as _cast

        mlx_trim_prompt_cache(_cast(list[object], self._draft_cache), n_positions)  # type: ignore[reportArgumentType]

    def reset_and_prefill(self, prompt_tokens: list[int]) -> None:
        """No-op: the legacy in-process path manages drafter cache externally.

        ``mlx_generate`` allocates the drafter cache, runs
        :func:`exo.worker.engines.mlx.generator.generate._spec_drafter_prefill`,
        and only then constructs this transport. Re-running prefill
        here would double-fill the cache. The Protocol method exists
        for symmetry with :class:`RemoteTransport`, where the drafter
        cache lives on the drafter rank and the transport owns its
        per-request reset/prefill.
        """
        del prompt_tokens

    def shutdown(self) -> None:
        return

    # -- internals --------------------------------------------------------

    def _run_drafter_forwards(self, inputs: list[int], num_forwards: int) -> list[int]:
        """Mirror of mlx_lm's ``_draft_generate`` semantics.

        Forward 0 consumes ``inputs`` (length 1 or 2); forwards 1..N-1
        consume the previous forward's sampled output. ``mx.async_eval``
        between forwards lets the GPU pipeline the dispatches.
        """
        ys: list[mx.array] = []
        y = mx.array(inputs, dtype=mx.uint32)
        for _ in range(num_forwards):
            logits = self._draft_model(y[None], cache=self._draft_cache)
            sampled = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
            mx.async_eval(sampled)
            ys.append(sampled)
            y = sampled
        # Force a sync at the end so the cache state is realised before
        # the spec loop dispatches target verify on top of these outputs.
        mx.eval(ys + [c.state for c in self._draft_cache])  # type: ignore[reportArgumentType]
        return [int(t.item()) for t in ys]


# ---------------------------------------------------------------------------
# Transport kind selection
# ---------------------------------------------------------------------------


ALL_TRANSPORT_KINDS: Final[tuple[str, ...]] = ("inprocess",)
"""Recognised values of ``EXO_DRAFTER_TRANSPORT``.

The ``"remote"`` option was retired alongside the ``mx.distributed``-
backed drafter wire (the v3+ asymmetric path uses a builder-supplied
:class:`RemoteTransport` bound to a TCP socket; it cannot be
constructed via this env-var factory because the socket comes from
target rank 0's listener and isn't available at process startup).
"""

EXO_DRAFTER_TRANSPORT_ENV: Final[str] = "EXO_DRAFTER_TRANSPORT"


@runtime_checkable
class HasNumDraftTokens(Protocol):
    """Anything exposing the per-transport wire-protocol K budget.

    Both :class:`DrafterTransport` (per-request session handles) and
    :class:`RemoteTransport` (long-lived target-side session factory)
    expose ``num_draft_tokens``. The clamp helper only ever needs that
    one number, so widen the parameter type to this Protocol so the
    clamp applies to *whichever* transport-shaped object the call site
    happens to have. Pre-fix the clamp accepted only
    ``DrafterTransport`` and silently skipped on
    ``RemoteTransport`` (the production asymmetric placement type),
    so oversized per-request ``num_draft_tokens`` survived all the way
    to ``forward(...)`` and crashed the request with ``ValueError``
    instead of being clamped (Codex P1, PR #20 round 5,
    generate.py:1025).
    """

    @property
    def num_draft_tokens(self) -> int: ...


def clamp_num_draft_tokens_to_transport(
    requested_num_draft_tokens: int,
    transport: HasNumDraftTokens,
) -> tuple[int, bool]:
    """Clamp a per-request K against the transport's wire-protocol budget.

    Asymmetric placement allocates a long-lived ``RemoteTransport`` at
    builder time with a fixed ``num_draft_tokens`` budget (see
    ``builder.py``). A per-request ``num_draft_tokens`` override above
    the budget would otherwise raise ``ValueError`` deep inside
    :class:`PipelinedModelDrafter`, killing the runner subprocess and
    leaving the peer rank wedged (regression: aborted K=8 sweep at
    14:35:05 took the target rank with it). Clamping silently to the
    transport max is the only safe behaviour: the wire-protocol budget
    is a startup-time setting (``EXO_NUM_DRAFT_TOKENS``) and cannot be
    widened mid-flight without re-warmup.

    Accepts any object exposing ``num_draft_tokens``: in production
    asymmetric placement this is :class:`RemoteTransport` (a
    session-factory, not a per-request session); in the in-process
    path it is the :class:`DrafterTransport` itself.

    Returns the (possibly clamped) K and a flag indicating whether
    clamping was applied so callers can emit a structured warning.

    :raises ValueError: if ``requested_num_draft_tokens`` is < 1. The
        spec loop never proposes zero or negative drafts, so this would
        be a programmer error rather than a malformed request.
    """
    if requested_num_draft_tokens < 1:
        raise ValueError(
            f"requested_num_draft_tokens must be >= 1, got {requested_num_draft_tokens}"
        )
    transport_max = transport.num_draft_tokens
    if requested_num_draft_tokens > transport_max:
        return transport_max, True
    return requested_num_draft_tokens, False


def parse_transport_kind(raw: str | None, default: str) -> str:
    """Parse the ``EXO_DRAFTER_TRANSPORT`` env var, warning on unknown values."""
    if raw is None:
        return default
    candidate = raw.strip().lower()
    if candidate in ALL_TRANSPORT_KINDS:
        return candidate
    # Imported lazily so this module is importable without the runner
    # bootstrap (used by tests that exercise the parser in isolation).
    from exo.worker.runner.bootstrap import logger

    logger.warning(
        f"{EXO_DRAFTER_TRANSPORT_ENV}={raw!r} not in {ALL_TRANSPORT_KINDS}; "
        f"falling back to {default!r}"
    )
    return default


def make_inprocess_transport(
    *,
    draft_model: Model | None,
    draft_cache: KVCacheType | None,
    num_draft_tokens: int,
    group: mx.distributed.Group | None = None,
    drafter_rank: int | None = None,
    target_rank: int | None = None,
) -> DrafterTransport:
    """Build an :class:`InProcessTransport`.

    Wrapped in a factory so callers don't import the concrete class;
    keeps the spec loop coupled only to the Protocol. The ``group`` /
    ``drafter_rank`` / ``target_rank`` kwargs are accepted (ignored) for
    parity with :func:`make_remote_transport`, so :func:`make_drafter`
    can dispatch to either factory with one call shape.
    """
    del group, drafter_rank, target_rank  # remote-only knobs
    if draft_model is None or draft_cache is None:
        raise ValueError(
            "InProcessTransport requires draft_model and draft_cache; "
            "remote transport is the only path that runs without them"
        )
    return InProcessTransport(
        draft_model=draft_model,
        draft_cache=draft_cache,
        num_draft_tokens=num_draft_tokens,
    )


# The dispatch table returns either a :class:`DrafterTransport` (in-process,
# directly consumable by the spec loop) or a :class:`RemoteTransport`
# (the wire owner; callers must call ``open_session()`` to obtain a
# :class:`DrafterTransport` view per request). Callers route on the
# concrete return type rather than relying on a single Protocol.
_TransportFactory = Callable[..., object]


def transport_factory_for(kind: str) -> _TransportFactory:
    """Return the factory for the requested transport kind.

    Only ``"inprocess"`` is constructible via this factory; the
    asymmetric remote transport (``RemoteTransport``) is built
    directly from the runner bootstrap with a connected socket from
    target rank 0's drafter listener.

    Raises:
        ValueError: ``kind`` is not in :data:`ALL_TRANSPORT_KINDS`.
    """
    if kind == "inprocess":
        return make_inprocess_transport
    raise ValueError(f"Unknown drafter transport kind: {kind!r}")


__all__ = [
    "ALL_TRANSPORT_KINDS",
    "DraftFuture",
    "DrafterTransport",
    "EXO_DRAFTER_TRANSPORT_ENV",
    "InProcessTransport",
    "make_inprocess_transport",
    "parse_transport_kind",
    "transport_factory_for",
]
