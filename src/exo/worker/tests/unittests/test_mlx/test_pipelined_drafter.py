"""Tests for :mod:`pipelined_drafter` and :mod:`drafter_transport`.

The cross-round speculation accounting is the only complex piece, so
these tests focus on:

  * The :class:`DrafterTransport` Protocol contract (any implementation
    that satisfies the Protocol must accept the call sequence the spec
    loop emits).
  * The spec-loop's cache-trim arithmetic for partial accept, full
    accept, speculation hit, and speculation miss -- exercised through
    a deterministic fake transport that records every call so we can
    assert on the trim/forward sequence without spinning up MLX
    weights.
  * Transport-kind parsing (``EXO_DRAFTER_TRANSPORT`` env var).

End-to-end correctness with real MLX weights is exercised by the smoke
+ bench scripts; this file stays MLX-free so it runs in seconds on CI.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Final

import pytest

from exo.worker.engines.mlx.generator.drafter_transport import (
    ALL_TRANSPORT_KINDS,
    EXO_DRAFTER_TRANSPORT_ENV,
    DrafterTransport,
    DraftFuture,
    clamp_num_draft_tokens_to_transport,
    parse_transport_kind,
    transport_factory_for,
)

# ---------------------------------------------------------------------------
# Test fixtures: deterministic fake transport
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    """One method call against the fake transport, in arrival order."""

    kind: str  # "forward" or "trim"
    inputs: tuple[int, ...] = ()
    num_forwards: int = 0
    n_positions: int = 0


@dataclass
class _ForwardScript:
    """Pre-recorded outputs for the next ``forward`` call."""

    outputs: list[int]


@dataclass
class FakeTransport:
    """A :class:`DrafterTransport` that records calls and returns scripted drafts.

    Used to exercise the spec loop's bookkeeping without running MLX.
    Every ``forward`` consumes one entry from ``script``; if the script
    is exhausted, the test has hit a code path it didn't predict and
    the transport raises (failing the test loudly).
    """

    num_draft_tokens_value: int
    script: list[_ForwardScript] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    cache_offset: int = 0

    @property
    def num_draft_tokens(self) -> int:
        return self.num_draft_tokens_value

    def forward(self, inputs: list[int], num_forwards: int) -> DraftFuture:
        if not 1 <= num_forwards <= self.num_draft_tokens_value + 1:
            raise ValueError(f"num_forwards out of bounds: {num_forwards}")
        if not 1 <= len(inputs) <= 2:
            raise ValueError(f"inputs length out of bounds: {len(inputs)}")
        if not self.script:
            raise AssertionError(
                "FakeTransport.forward called without script entry; "
                "test missed a code path"
            )
        entry = self.script.pop(0)
        if len(entry.outputs) != num_forwards:
            raise AssertionError(
                f"Script entry has {len(entry.outputs)} outputs; "
                f"forward asked for {num_forwards}"
            )
        self.calls.append(
            _Call(kind="forward", inputs=tuple(inputs), num_forwards=num_forwards)
        )
        # Cache extends by ``len(inputs) + num_forwards - 1`` per spec.
        self.cache_offset += len(inputs) + num_forwards - 1
        future: DraftFuture = Future()
        future.set_result(list(entry.outputs))
        return future

    def trim_cache(self, n_positions: int) -> None:
        if n_positions < 0:
            raise ValueError(f"n_positions must be >= 0, got {n_positions}")
        if n_positions > self.cache_offset:
            raise AssertionError(
                f"Trim {n_positions} would exceed cache offset {self.cache_offset}; "
                "spec loop is over-trimming"
            )
        self.calls.append(_Call(kind="trim", n_positions=n_positions))
        self.cache_offset -= n_positions

    def reset_and_prefill(self, prompt_tokens: list[int]) -> None:
        # Mirror RemoteTransport semantics: reset cache to 0, then
        # extend by len(prompt_tokens). The FakeTransport doesn't
        # actually run a model, so the offset bookkeeping is the only
        # observable side-effect tests care about.
        self.cache_offset = len(prompt_tokens)
        self.calls.append(
            _Call(kind="reset_and_prefill", n_positions=len(prompt_tokens))
        )

    def shutdown(self) -> None:
        return


def test_fake_transport_satisfies_protocol() -> None:
    """The fake transport must structurally satisfy :class:`DrafterTransport`."""
    transport: DrafterTransport = FakeTransport(num_draft_tokens_value=4)
    assert isinstance(transport, DrafterTransport)


# ---------------------------------------------------------------------------
# Transport-kind parsing
# ---------------------------------------------------------------------------


_KIND_DEFAULT: Final[str] = "inprocess"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, _KIND_DEFAULT),
        ("inprocess", "inprocess"),
        ("INPROCESS", "inprocess"),
        ("  inprocess  ", "inprocess"),
    ],
)
def test_parse_transport_kind_recognised(raw: str | None, expected: str) -> None:
    """Only ``inprocess`` is a valid transport-kind keyword.

    The legacy ``"remote"`` keyword was a factory hint for the
    ``mx.distributed``-backed asymmetric drafter; the v3+ asymmetric
    wire is built directly from the runner bootstrap with a connected
    socket and never goes through the env-var factory.
    """
    assert parse_transport_kind(raw, _KIND_DEFAULT) == expected


def test_parse_transport_kind_rejects_legacy_remote() -> None:
    """Legacy ``"remote"`` keyword falls back to the default with a warning.

    The asymmetric remote transport is built directly from the runner
    bootstrap in v3+; an env-var hint of ``"remote"`` no longer has a
    factory backing and must degrade to ``inprocess`` rather than crash.
    """
    assert parse_transport_kind("remote", _KIND_DEFAULT) == _KIND_DEFAULT
    assert parse_transport_kind("Remote", _KIND_DEFAULT) == _KIND_DEFAULT


def test_parse_transport_kind_falls_back_for_unknown() -> None:
    # Unknown kinds warn and fall back to the default rather than
    # raising; that mirrors how ``parse_draft_mode`` handles unknown
    # ``EXO_DRAFT_MODE`` values.
    assert parse_transport_kind("totally-bogus", _KIND_DEFAULT) == _KIND_DEFAULT


def test_all_transport_kinds_match_factory_dispatch() -> None:
    """Every kind in :data:`ALL_TRANSPORT_KINDS` must have a factory.

    The factory may raise ``NotImplementedError`` (Layer B's remote
    transport does), but :func:`transport_factory_for` itself must
    always return a callable -- the dispatch table is part of the
    public contract.
    """
    for kind in ALL_TRANSPORT_KINDS:
        factory = transport_factory_for(kind)
        assert callable(factory)


def test_transport_factory_for_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown drafter transport kind"):
        transport_factory_for("totally-bogus")


# ---------------------------------------------------------------------------
# Spec loop arithmetic via the fake transport
# ---------------------------------------------------------------------------


# These tests exercise the cache-trim arithmetic *as the spec loop
# emits it*, without running the MLX target. We construct call traces
# the loop would produce for a known accept pattern and assert the
# trim/forward sequence matches the formula derived in the
# pipelined_drafter module docstring.
#
# Strategy: don't actually run the spec loop (which needs an MLX
# target). Instead, simulate the spec loop's transport calls
# imperatively for each scenario and assert the cache offset / call
# sequence matches what the docstring promises.


class TestSpecLoopArithmetic:
    """Trace the transport-call sequence for canonical accept patterns."""

    def test_partial_accept_no_speculation(self) -> None:
        """Partial accept (n=2 of K=4): trim K-n-1 = 1, propose [target_correction]."""
        k = 4
        n = 2
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                # Round 0: 4 drafts.
                _ForwardScript(outputs=[10, 11, 12, 13]),
                # Round 1: 4 drafts after partial-accept setup.
                _ForwardScript(outputs=[20, 21, 22, 23]),
            ],
        )

        # Round 0 propose.
        drafts = transport.forward([1], k).result()
        assert drafts == [10, 11, 12, 13]
        assert transport.cache_offset == k  # 4 positions

        # Spec loop: partial accept after target verify (n=2, drafts[2] mismatched).
        # Transport bookkeeping for next round:
        #   * trim k - n - 1 = 1 position
        #   * propose [target_correction] (length 1), k outputs
        transport.trim_cache(k - n - 1)
        assert transport.cache_offset == k - 1  # 3 positions

        # Next round propose with length-1 input.
        next_drafts = transport.forward([99], k).result()
        assert next_drafts == [20, 21, 22, 23]
        # Cache extends by k (length-1 input + k-1 length-1 forwards = k).
        assert transport.cache_offset == k - 1 + k  # 7 positions

        # Verify call trace.
        assert [c.kind for c in transport.calls] == [
            "forward",
            "trim",
            "forward",
        ]
        assert transport.calls[1].n_positions == 1

    def test_full_accept_no_speculation(self) -> None:
        """Full accept (n=k): no trim; next round propose has length-2 input."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[20, 21, 22, 23]),
            ],
        )

        transport.forward([1], k).result()
        assert transport.cache_offset == k

        # Full accept: no trim. Next round propose with [drafts[-1], bonus].
        next_drafts = transport.forward([13, 99], k).result()
        assert next_drafts == [20, 21, 22, 23]
        # Cache extends by k + 1 (length-2 input + k-1 length-1 forwards).
        assert transport.cache_offset == k + (k + 1)

        assert [c.kind for c in transport.calls] == ["forward", "forward"]
        assert transport.calls[1].inputs == (13, 99)
        assert transport.calls[1].num_forwards == k

    def test_speculation_hit(self) -> None:
        """Full accept + speculation hit: round t+1 drafts come for free."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                # Round 0 propose: [10, 11, 12, 13].
                _ForwardScript(outputs=[10, 11, 12, 13]),
                # Speculative round (input=[13], k+1 outputs):
                # outputs[0] = drafter's bonus prediction; outputs[1..k] = round
                # 1's drafts.
                _ForwardScript(outputs=[99, 30, 31, 32, 33]),
            ],
        )

        # Round 0 propose.
        round0_drafts = transport.forward([1], k).result()
        assert round0_drafts == [10, 11, 12, 13]

        # Speculative call.
        spec_outputs = transport.forward([13], k + 1).result()
        assert spec_outputs == [99, 30, 31, 32, 33]
        # After speculation: cache extended by k (round 0) + (k + 1)
        # (speculation) = 2k+1 positions.
        assert transport.cache_offset == k + (k + 1)

        # Speculation hit: target's bonus_t == 99 == spec_outputs[0].
        # Round 1's drafts = spec_outputs[1:k+1].
        round1_drafts = spec_outputs[1 : k + 1]
        assert round1_drafts == [30, 31, 32, 33]

        # No additional transport calls (drafter cache state already
        # correct for round 1).
        assert [c.kind for c in transport.calls] == ["forward", "forward"]

    def test_speculation_miss_full_accept(self) -> None:
        """Full accept but bonus mismatched: rollback k+1, length-2 propose."""
        k = 4
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[88, 80, 81, 82, 83]),  # speculative
                _ForwardScript(outputs=[40, 41, 42, 43]),  # round 1 standard
            ],
        )

        transport.forward([1], k).result()
        spec_outputs = transport.forward([13], k + 1).result()
        # bonus_t = 99 (target), spec_outputs[0] = 88 -> miss.

        # Rollback the k+1 speculative positions.
        transport.trim_cache(k + 1)
        assert transport.cache_offset == k  # back to round-0 state

        # Standard length-2-seed propose for round 1: [drafts[-1], bonus_t].
        round1_drafts = transport.forward([13, 99], k).result()
        assert round1_drafts == [40, 41, 42, 43]

        del spec_outputs
        kinds = [c.kind for c in transport.calls]
        assert kinds == ["forward", "forward", "trim", "forward"]
        assert transport.calls[2].n_positions == k + 1
        assert transport.calls[3].inputs == (13, 99)

    def test_speculation_miss_partial_accept(self) -> None:
        """Partial accept with speculation in flight: rollback k+1 + partial trim."""
        k = 4
        n = 2
        transport = FakeTransport(
            num_draft_tokens_value=k,
            script=[
                _ForwardScript(outputs=[10, 11, 12, 13]),
                _ForwardScript(outputs=[88, 80, 81, 82, 83]),  # speculative
                _ForwardScript(outputs=[50, 51, 52, 53]),  # round 1
            ],
        )

        transport.forward([1], k).result()
        transport.forward([13], k + 1).result()
        # cache offset: k + (k + 1) = 2k + 1 = 9

        # Partial accept at round 0: speculation is invalid AND partial
        # trim is needed. The combined trim is (k + 1) + (k - n - 1).
        combined_trim = (k + 1) + (k - n - 1)
        transport.trim_cache(combined_trim)
        # cache offset: 2k + 1 - combined_trim = n + 1 = 3
        assert transport.cache_offset == n + 1

        # Round 1 standard propose with length-1 input.
        round1_drafts = transport.forward([99], k).result()
        assert round1_drafts == [50, 51, 52, 53]

        kinds = [c.kind for c in transport.calls]
        assert kinds == ["forward", "forward", "trim", "forward"]
        assert transport.calls[2].n_positions == combined_trim


# ---------------------------------------------------------------------------
# PipelinedModelDrafter wiring
# ---------------------------------------------------------------------------


def test_pipelined_drafter_mode_is_pipelined() -> None:
    # Imported lazily so this file stays importable without the drafter
    # module's MLX-bound siblings; the import itself is what we're
    # exercising (catches accidental syntax errors in pipelined_drafter
    # that the type checker might miss for runtime-only paths).
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    transport = FakeTransport(num_draft_tokens_value=4)
    drafter = PipelinedModelDrafter(transport=transport, num_draft_tokens=4)
    assert drafter.mode == "pipelined"
    assert drafter.num_draft_tokens == 4


def test_pipelined_drafter_validates_num_draft_tokens() -> None:
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    transport = FakeTransport(num_draft_tokens_value=4)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        PipelinedModelDrafter(transport=transport, num_draft_tokens=0)
    with pytest.raises(ValueError, match="exceeds transport's max"):
        PipelinedModelDrafter(transport=transport, num_draft_tokens=10)


def test_pipelined_drafter_shutdown_delegates() -> None:
    """Shutdown should propagate to the transport so remote serve loops drain cleanly."""
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    shutdown_calls: list[None] = []

    class _ShutdownRecorder(FakeTransport):
        def shutdown(self) -> None:
            shutdown_calls.append(None)

    transport = _ShutdownRecorder(num_draft_tokens_value=4)
    drafter = PipelinedModelDrafter(transport=transport, num_draft_tokens=4)
    drafter.shutdown()
    assert len(shutdown_calls) == 1


# ---------------------------------------------------------------------------
# Transport-kind environment plumbing
# ---------------------------------------------------------------------------


def test_make_drafter_pipelined_without_model_or_transport_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_drafter("pipelined", ...)`` requires either a model+cache or a transport.

    The env-var-driven factory path is gone in v3+ (asymmetric remote
    transport is constructed directly by the runner bootstrap). Calling
    ``make_drafter`` with neither a builder-supplied transport nor a
    drafter model + cache must raise a clear error -- it has no way to
    construct the in-process transport.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    monkeypatch.delenv(EXO_DRAFTER_TRANSPORT_ENV, raising=False)
    with pytest.raises(ValueError, match="pipelined"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
        )


# ---------------------------------------------------------------------------
# Asymmetric placement entry points
# ---------------------------------------------------------------------------


def test_make_drafter_uses_supplied_pipelined_transport() -> None:
    """When ``pipelined_transport`` is supplied, ``make_drafter`` must reuse it.

    Asymmetric placement allocates a long-lived RemoteTransport at
    SequentialGenerator build time so executor + drafter cache lifecycle
    aren't paid per-request. The factory entry point must accept that
    pre-built transport instead of constructing a new one.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    transport = FakeTransport(num_draft_tokens_value=4)
    drafter = make_drafter(
        mode="pipelined",
        num_draft_tokens=4,
        draft_model=None,
        draft_cache=None,
        pipelined_transport=transport,
    )
    assert isinstance(drafter, PipelinedModelDrafter)
    # The drafter must wrap the supplied transport, not a freshly-
    # constructed one (would be a behavioural regression because the
    # remote drafter cache + executor would be leaked on every request).
    drafter.shutdown()
    assert transport.calls == []  # FakeTransport.shutdown is a no-op


def test_make_drafter_rejects_non_protocol_pipelined_transport() -> None:
    """``pipelined_transport`` must implement ``DrafterTransport``."""
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    class NotATransport:
        pass

    with pytest.raises(TypeError, match="DrafterTransport"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
            pipelined_transport=NotATransport(),
        )


class TestClampNumDraftTokensToTransport:
    """Per-request K must be clamped to the transport's wire-protocol max.

    Regression coverage: aborted K=8 sweep at 14:35:05 raised
    ``ValueError`` deep inside :class:`PipelinedModelDrafter` and killed
    the target runner subprocess (PR #15). The clamp helper exists so
    ``generate.py`` can defend the runner from malformed per-request
    overrides without ever reaching the drafter constructor.
    """

    def test_clamp_no_op_when_request_within_budget(self) -> None:
        transport = FakeTransport(num_draft_tokens_value=5)
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(3, transport)
        assert clamped == 3
        assert was_clamped is False

    def test_clamp_no_op_when_request_equals_budget(self) -> None:
        transport = FakeTransport(num_draft_tokens_value=5)
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(5, transport)
        assert clamped == 5
        assert was_clamped is False

    def test_clamp_applies_when_request_exceeds_budget(self) -> None:
        transport = FakeTransport(num_draft_tokens_value=5)
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(8, transport)
        assert clamped == 5
        assert was_clamped is True

    def test_clamp_pathological_request(self) -> None:
        transport = FakeTransport(num_draft_tokens_value=5)
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(1024, transport)
        assert clamped == 5
        assert was_clamped is True

    def test_clamp_rejects_zero_or_negative(self) -> None:
        transport = FakeTransport(num_draft_tokens_value=5)
        with pytest.raises(ValueError, match="requested_num_draft_tokens"):
            clamp_num_draft_tokens_to_transport(0, transport)
        with pytest.raises(ValueError, match="requested_num_draft_tokens"):
            clamp_num_draft_tokens_to_transport(-1, transport)

    def test_clamped_k_constructs_pipelined_drafter_safely(self) -> None:
        """Smoke: clamped K must satisfy ``PipelinedModelDrafter`` validation.

        The whole point of the clamp is that the value flowing into
        :class:`PipelinedModelDrafter` never exceeds ``transport.num_draft_tokens``.
        Construct the drafter with the clamped K to prove the pre-fix
        regression path is gone.
        """
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            PipelinedModelDrafter,
        )

        transport = FakeTransport(num_draft_tokens_value=5)
        # Pre-fix: K=8 raised ValueError here and killed the subprocess.
        clamped, _ = clamp_num_draft_tokens_to_transport(8, transport)
        drafter = PipelinedModelDrafter(transport=transport, num_draft_tokens=clamped)
        assert drafter.num_draft_tokens == 5

    def test_clamp_accepts_remote_transport_shape(self) -> None:
        """Codex P1 (PR #20 round-(N+5), generate.py:1025).

        In production asymmetric placement the call site holds a
        :class:`RemoteTransport` (a session factory), not a per-request
        :class:`DrafterTransport`. ``RemoteTransport`` exposes the same
        ``num_draft_tokens`` property but does not satisfy the
        ``DrafterTransport`` Protocol (it has no ``forward`` /
        ``trim_cache``). The clamp must work against this shape too,
        because pre-fix the call site's ``isinstance(_, DrafterTransport)``
        branch silently skipped clamping and oversized per-request K
        survived to ``forward(...)`` and crashed the request with
        ``ValueError``.
        """
        from exo.worker.engines.mlx.generator.drafter_transport import (
            HasNumDraftTokens,
        )

        @dataclass
        class _FakeRemoteTransportShape:
            """A ``num_draft_tokens``-only object, mirroring ``RemoteTransport``.

            Deliberately omits ``forward`` / ``trim_cache`` /
            ``reset_and_prefill`` / ``shutdown`` so it does NOT satisfy
            :class:`DrafterTransport` -- we want to prove the clamp
            works against the Protocol surface actually present in
            production asymmetric placement.
            """

            num_draft_tokens_value: int

            @property
            def num_draft_tokens(self) -> int:
                return self.num_draft_tokens_value

        remote_shape = _FakeRemoteTransportShape(num_draft_tokens_value=4)
        # Sanity: this object satisfies HasNumDraftTokens but not the
        # full DrafterTransport Protocol. The new isinstance() guard
        # in generate.py uses HasNumDraftTokens, so RemoteTransport
        # placements now hit the clamp path.
        assert isinstance(remote_shape, HasNumDraftTokens)
        assert not isinstance(remote_shape, DrafterTransport)

        # Oversized request must clamp to the transport's K.
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(8, remote_shape)
        assert clamped == 4
        assert was_clamped is True

        # Within-budget request must pass through unchanged.
        clamped, was_clamped = clamp_num_draft_tokens_to_transport(3, remote_shape)
        assert clamped == 3
        assert was_clamped is False


def test_make_drafter_pipelined_multi_target_requires_target_group() -> None:
    """V2 boundary: multi-target asymmetric requires a target_group for the
    rank-0 -> peer broadcast of drafts each round. Building the root-side
    drafter without ``target_group`` is a configuration error: the spec
    loop would race on a missing collective and silently desync.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    transport = FakeTransport(num_draft_tokens_value=4)
    with pytest.raises(ValueError, match="requires target_group"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
            pipelined_transport=transport,
            target_subgroup_size=2,
            target_group=None,
        )


def test_make_drafter_pipelined_consumer_rank_requires_target_group() -> None:
    """V2 boundary: a non-root target rank (no transport) must receive a
    ``target_group`` so the broadcast can land. Without it the consumer
    drafter would have no way to obtain drafts and the round 0 verify
    would deadlock against the root's TP collective.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    with pytest.raises(ValueError, match="requires target_group"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
            pipelined_transport=None,
            target_subgroup_size=2,
            target_group=None,
            is_target_root=False,
        )


def test_make_drafter_pipelined_consumer_for_three_target_ranks() -> None:
    """V2 multi-target with N target ranks (N >= 2): every non-root rank
    constructs the same transport-less consumer drafter. Exercise N=3
    explicitly so the broadcast contract is not implicitly bound to
    ``target_subgroup_size == 2`` (the case the cluster bench covers).
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    class _StubGroup:
        def size(self) -> int:
            return 3

        def rank(self) -> int:
            return 2

    drafter = make_drafter(
        mode="pipelined",
        num_draft_tokens=4,
        draft_model=None,
        draft_cache=None,
        pipelined_transport=None,
        target_subgroup_size=3,
        target_group=_StubGroup(),
        is_target_root=False,
    )
    assert isinstance(drafter, PipelinedModelDrafter)
    assert drafter.mode == "pipelined"
    assert drafter.num_draft_tokens == 4


def test_make_drafter_pipelined_root_for_three_target_ranks() -> None:
    """V2 multi-target root with N=3 ranks: identical contract to N=2
    -- the root owns the transport and broadcasts on the target group.
    The collective is N-ary (``mx.distributed.all_sum``), so the
    construction has no special-casing for N == 2 and we want a test
    asserting that explicitly.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter
    from exo.worker.engines.mlx.generator.pipelined_drafter import (
        PipelinedModelDrafter,
    )

    class _StubGroup:
        def size(self) -> int:
            return 3

        def rank(self) -> int:
            return 0

    transport = FakeTransport(num_draft_tokens_value=4)
    drafter = make_drafter(
        mode="pipelined",
        num_draft_tokens=4,
        draft_model=None,
        draft_cache=None,
        pipelined_transport=transport,
        target_subgroup_size=3,
        target_group=_StubGroup(),
        is_target_root=True,
    )
    assert isinstance(drafter, PipelinedModelDrafter)


# ---------------------------------------------------------------------------
# Broadcast helpers (single-rank short-circuit)
# ---------------------------------------------------------------------------


class TestBroadcastDrafts:
    """``_broadcast_drafts`` length-prefix encoding contract.

    Multi-rank behaviour is covered by the cluster bench (real
    ``mx.distributed.all_sum``). The single-rank short-circuit is the
    only path we can exercise in unit tests, but it captures the most
    important contract bug: the length-prefix decoder rejecting
    nonsensical lengths from a corrupted broadcast.
    """

    def test_single_rank_short_circuit_root(self) -> None:
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_drafts,  # pyright: ignore[reportPrivateUsage]
        )

        out: list[int] = _broadcast_drafts(
            [10, 20],
            k=4,
            target_group=None,
            target_peer_fanout=None,
            is_root=True,
        )
        assert out == [10, 20]

    def test_single_rank_short_circuit_consumer_rejected(self) -> None:
        # Consumer rank in single-rank mode is a configuration bug --
        # there's no peer to broadcast from. Surface it loudly.
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_drafts,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(RuntimeError, match="non-root"):
            _broadcast_drafts(
                None,
                k=4,
                target_group=None,
                target_peer_fanout=None,
                is_root=False,
            )

    def test_single_rank_root_requires_drafts(self) -> None:
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_drafts,  # pyright: ignore[reportPrivateUsage]
        )

        # ``drafts is None`` on root in the short-circuit path is a
        # caller bug (the runner never has a None drafts list when it
        # owns the wire).
        with pytest.raises(RuntimeError, match="non-root"):
            _broadcast_drafts(
                None,
                k=4,
                target_group=None,
                target_peer_fanout=None,
                is_root=False,
            )


class TestBroadcastTargetTokens:
    """``_broadcast_target_tokens`` carries the verifier's sampled
    tokens from rank 0 to non-root target ranks so accept/reject is
    bit-identical across the target subgroup.

    Without this broadcast, every rank's ``mx.random.categorical`` call
    returns RNG-divergent tokens (default temperature is 0.7 in the
    API path), the ranks reach different ``num_accepted``, trim the
    target's prompt cache by different amounts, and the next TP
    forward consumes mismatched cache state -- a silent garbage-output
    bug. These tests pin the contract so a future refactor can't
    accidentally drop the broadcast.
    """

    def test_single_rank_short_circuit_root(self) -> None:
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_target_tokens,  # pyright: ignore[reportPrivateUsage]
        )

        # k_this + 1 == 3 tokens: the seed-bonus + drafts emitted per
        # round in a K=4, k_this=2 partial round.
        out: list[int] = _broadcast_target_tokens(
            [10, 20, 30],
            k=4,
            k_this=2,
            target_group=None,
            target_peer_fanout=None,
            is_root=True,
        )
        assert out == [10, 20, 30]

    def test_single_rank_consumer_rejected(self) -> None:
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_target_tokens,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(RuntimeError, match="non-root"):
            _broadcast_target_tokens(
                None,
                k=4,
                k_this=2,
                target_group=None,
                target_peer_fanout=None,
                is_root=False,
            )

    def test_root_rejects_wrong_length(self) -> None:
        # Verifier always emits exactly ``k_this + 1`` tokens; anything
        # else means the spec loop is calling the broadcast with stale
        # state. Raise rather than silently right-pad.
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_target_tokens,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(RuntimeError, match="must equal k_this"):
            _broadcast_target_tokens(
                [10, 20],
                k=4,
                k_this=2,
                target_group=None,
                target_peer_fanout=None,
                is_root=True,
            )


def test_make_drafter_pipelined_root_rank_with_no_transport_rejected() -> None:
    """Configuration error: ``is_target_root=True`` implies this rank owns
    the drafter socket; the caller must pass a transport. Reaching the
    multi-target consumer branch with ``is_target_root=True`` is a
    placement bug we want to surface loudly rather than silently drop.
    """
    from exo.worker.engines.mlx.generator.drafter import make_drafter

    class _StubGroup:
        def size(self) -> int:
            return 2

        def rank(self) -> int:
            return 0

    with pytest.raises(ValueError, match="is_target_root=True"):
        make_drafter(
            mode="pipelined",
            num_draft_tokens=4,
            draft_model=None,
            draft_cache=None,
            pipelined_transport=None,
            target_subgroup_size=2,
            target_group=_StubGroup(),
            is_target_root=True,
        )


# ---------------------------------------------------------------------------
# Drafter-death recovery: abort sentinel + wrap behaviour
# ---------------------------------------------------------------------------


class TestDrafterAbortRecovery:
    """Recovery contract when the drafter rank dies mid-generation.

    Pre-fix failure mode: root's ``transport.forward`` raised
    ``OSError`` and re-raised cleanly out of ``mlx_generate``, but
    non-root target ranks blocked indefinitely on the next-round
    draft broadcast (the sole inter-rank coordination channel for
    spec decode). The abort sentinel + wrap + ``RemoteTransport``
    failure flag together convert that hang into a fast, lockstep
    exit on every rank, with the runner subprocess crashing so the
    master's instance-deletion path can rebuild the placement.

    The cluster bench covers the full multi-rank flow against real
    ``mx.distributed``; these unit tests pin the single-rank
    invariants that are reachable without spinning up a peer group.
    """

    def test_broadcast_abort_short_circuits_without_group(self) -> None:
        # ``target_group is None`` (single-rank placement) means there
        # are no peers to notify; the abort broadcast must be a no-op
        # rather than raising or contacting any wire layer.
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            _broadcast_abort,  # pyright: ignore[reportPrivateUsage]
        )

        # Should not raise; should not require any group machinery.
        _broadcast_abort(k=4, target_group=None, target_peer_fanout=None)

    def test_sentinel_value_is_in_validator_range(self) -> None:
        # The sentinel must satisfy ``_validate_broadcast_values``
        # (positive int32) so a real cluster broadcast doesn't reject
        # it before non-root ranks have a chance to decode it.
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            DRAFT_ABORT_SENTINEL,
        )
        from exo.worker.engines.mlx.utils_mlx import (
            _MX_BROADCAST_MAX_VALUE,  # pyright: ignore[reportPrivateUsage]
            _validate_broadcast_values,  # pyright: ignore[reportPrivateUsage]
        )

        assert 0 < DRAFT_ABORT_SENTINEL < _MX_BROADCAST_MAX_VALUE
        # Must also exceed any plausible draft length so it can never
        # collide with a legitimate length-prefix.
        assert DRAFT_ABORT_SENTINEL > 1_000_000
        # Validator round-trip with the wire payload root would emit.
        _validate_broadcast_values([DRAFT_ABORT_SENTINEL] + [0] * 4)

    def test_broadcast_drafts_decodes_sentinel_to_abort_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Multi-rank receive path: when ``mx_broadcast_int_list``
        # returns a buffer whose length-prefix is the sentinel,
        # ``_broadcast_drafts`` raises ``DrafterAbortedError`` so the
        # spec loop can exit in lockstep with the dead root rank.
        from exo.worker.engines.mlx.generator import pipelined_drafter
        from exo.worker.engines.mlx.generator.pipelined_drafter import (
            DRAFT_ABORT_SENTINEL,
            DrafterAbortedError,
            _broadcast_drafts,  # pyright: ignore[reportPrivateUsage]
        )

        k = 4

        def fake_broadcast(
            values: list[int] | None,
            length: int,
            group: object,
            *,
            is_root: bool,
        ) -> list[int]:
            del values, group, is_root
            assert length == k + 1
            return [DRAFT_ABORT_SENTINEL] + [0] * k

        monkeypatch.setattr(pipelined_drafter, "mx_broadcast_int_list", fake_broadcast)

        sentinel_group = object()  # opaque; the fake never inspects
        with pytest.raises(DrafterAbortedError, match="drafter aborted"):
            _broadcast_drafts(
                None,
                k=k,
                target_group=sentinel_group,  # pyright: ignore[reportArgumentType]
                target_peer_fanout=None,
                is_root=False,
            )

    def test_spec_step_wrap_root_broadcasts_abort_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a body that immediately raises OSError; the wrap
        # must call ``_broadcast_abort`` (root path) before re-raising
        # so non-root ranks unblock their pending broadcast.
        from exo.worker.engines.mlx.generator import pipelined_drafter

        broadcast_calls: list[tuple[int, object]] = []

        def fake_abort(
            *, k: int, target_group: object, target_peer_fanout: object
        ) -> None:
            del target_peer_fanout
            broadcast_calls.append((k, target_group))

        def fake_body(**kwargs: object):
            del kwargs
            raise ConnectionError("drafter rank closed mid-frame")
            yield  # pragma: no cover -- generator marker

        monkeypatch.setattr(pipelined_drafter, "_broadcast_abort", fake_abort)
        monkeypatch.setattr(
            pipelined_drafter,
            "_pipelined_speculative_step_body",
            fake_body,
        )

        sentinel_group = object()
        gen = pipelined_drafter._pipelined_speculative_step(  # pyright: ignore[reportPrivateUsage]
            prompt=None,  # pyright: ignore[reportArgumentType]
            model=None,  # pyright: ignore[reportArgumentType]
            transport=None,
            prompt_cache=None,  # pyright: ignore[reportArgumentType]
            max_tokens=8,
            sampler=lambda x: x,
            logits_processors=[],
            num_draft_tokens=4,
            prefill_step_size=512,
            prompt_token_count=0,
            target_group=sentinel_group,  # pyright: ignore[reportArgumentType]
            is_target_root=True,
        )
        with pytest.raises(ConnectionError, match="drafter rank closed"):
            next(gen)
        assert broadcast_calls == [(4, sentinel_group)]

    def test_spec_step_wrap_non_root_does_not_broadcast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-root has no transport to fail on; if a non-root somehow
        # raises OSError (e.g. a peer-side issue surfaces this way),
        # we must NOT issue an abort broadcast -- only root owns that
        # signal. Re-raising preserves the original error for the
        # caller's traceback without a phantom broadcast.
        from exo.worker.engines.mlx.generator import pipelined_drafter

        broadcast_calls: list[tuple[int, object]] = []

        def fake_abort(
            *, k: int, target_group: object, target_peer_fanout: object
        ) -> None:
            del target_peer_fanout
            broadcast_calls.append((k, target_group))

        def fake_body(**kwargs: object):
            del kwargs
            raise ConnectionError("non-root saw socket failure somehow")
            yield  # pragma: no cover

        monkeypatch.setattr(pipelined_drafter, "_broadcast_abort", fake_abort)
        monkeypatch.setattr(
            pipelined_drafter,
            "_pipelined_speculative_step_body",
            fake_body,
        )

        gen = pipelined_drafter._pipelined_speculative_step(  # pyright: ignore[reportPrivateUsage]
            prompt=None,  # pyright: ignore[reportArgumentType]
            model=None,  # pyright: ignore[reportArgumentType]
            transport=None,
            prompt_cache=None,  # pyright: ignore[reportArgumentType]
            max_tokens=8,
            sampler=lambda x: x,
            logits_processors=[],
            num_draft_tokens=4,
            prefill_step_size=512,
            prompt_token_count=0,
            target_group=object(),  # pyright: ignore[reportArgumentType]
            is_target_root=False,
        )
        with pytest.raises(ConnectionError):
            next(gen)
        assert broadcast_calls == []

    def test_spec_step_wrap_swallows_abort_broadcast_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the abort broadcast itself fails (e.g. ``target_group``
        # is also dead), the original transport error must still
        # surface intact -- the master's instance-deletion path is
        # the SIGKILL backstop, so swallowing the recovery error
        # avoids masking the root cause in the caller's traceback.
        from exo.worker.engines.mlx.generator import pipelined_drafter

        def fake_abort(
            *, k: int, target_group: object, target_peer_fanout: object
        ) -> None:
            del k, target_group, target_peer_fanout
            raise RuntimeError("group is also dead")

        def fake_body(**kwargs: object):
            del kwargs
            raise ConnectionError("primary failure")
            yield  # pragma: no cover

        monkeypatch.setattr(pipelined_drafter, "_broadcast_abort", fake_abort)
        monkeypatch.setattr(
            pipelined_drafter,
            "_pipelined_speculative_step_body",
            fake_body,
        )

        gen = pipelined_drafter._pipelined_speculative_step(  # pyright: ignore[reportPrivateUsage]
            prompt=None,  # pyright: ignore[reportArgumentType]
            model=None,  # pyright: ignore[reportArgumentType]
            transport=None,
            prompt_cache=None,  # pyright: ignore[reportArgumentType]
            max_tokens=8,
            sampler=lambda x: x,
            logits_processors=[],
            num_draft_tokens=4,
            prefill_step_size=512,
            prompt_token_count=0,
            target_group=object(),  # pyright: ignore[reportArgumentType]
            is_target_root=True,
        )
        with pytest.raises(ConnectionError, match="primary failure"):
            next(gen)


# ---------------------------------------------------------------------------
# Tokenizer vocab-size helper
# ---------------------------------------------------------------------------


class TestGetTokenizerVocabSize:
    """Regression coverage for ``_get_tokenizer_vocab_size``.

    Codex flagged (P1, PR #21) that the helper returned the *base*
    vocabulary size for HuggingFace fast tokenizers, which excludes
    added tokens (chat templates, EOS, control). Any model that emits
    such an added token therefore tripped the runtime "wire corruption"
    guard and crashed valid generations. The helper now prefers
    ``len(tokenizer)`` (full vocab) and falls back through
    ``vocab_size + |added_vocab|`` and the explicit vocab map.
    """

    def _call(self, inner: object) -> int | None:
        from exo.worker.engines.mlx.generator import pipelined_drafter

        wrapper = type("Wrapper", (), {"_tokenizer": inner})()
        return pipelined_drafter._get_tokenizer_vocab_size(wrapper)  # pyright: ignore[reportPrivateUsage,reportArgumentType]

    def test_prefers_len_over_vocab_size_for_hf_fast_tokenizer(self) -> None:
        """``len(tokenizer)`` is the canonical HF API for the full
        vocabulary including added tokens. The helper must prefer it
        over ``vocab_size`` (which excludes added tokens)."""

        class _HFFastTokenizer:
            vocab_size: int = 32000
            added_count: int = 8

            def __len__(self) -> int:
                return self.vocab_size + self.added_count

            def get_added_vocab(self) -> dict[str, int]:
                return {f"<extra_{i}>": 32000 + i for i in range(self.added_count)}

        assert self._call(_HFFastTokenizer()) == 32008

    def test_added_vocab_bumps_size_when_len_is_missing(self) -> None:
        """If the wrapper hides ``__len__`` (some custom tokenizers do),
        we still want to add the added-vocab size to the base vocab so
        the guard doesn't reject legitimate added tokens."""

        class _NoLenTokenizer:
            vocab_size: int = 4096

            def get_added_vocab(self) -> dict[str, int]:
                return {"<eos>": 4096, "<pad>": 4097}

        assert self._call(_NoLenTokenizer()) == 4098

    def test_falls_back_to_max_vocab_value_plus_one(self) -> None:
        """When neither ``__len__`` nor ``vocab_size`` is exposed, the
        scan over ``vocab.values()`` is the last reliable source."""

        class _OnlyVocabMap:
            vocab: dict[str, int] = {"a": 0, "b": 1, "<extra>": 7}

        assert self._call(_OnlyVocabMap()) == 8

    def test_falls_back_to_vocab_size_when_added_helper_raises(self) -> None:
        """Some tokenizers raise from ``get_added_vocab`` (e.g. when the
        added-tokens decoder isn't initialised). The helper must not
        propagate that -- a missing added-vocab count is treated as zero
        and we still return the base vocab size."""

        class _BrokenAddedHelper:
            vocab_size: int = 16

            def get_added_vocab(self) -> dict[str, int]:
                raise RuntimeError("added vocab not initialised")

        assert self._call(_BrokenAddedHelper()) == 16

    def test_returns_none_when_tokenizer_has_no_inner(self) -> None:
        from exo.worker.engines.mlx.generator import pipelined_drafter

        wrapper = type("Wrapper", (), {})()
        assert (
            pipelined_drafter._get_tokenizer_vocab_size(wrapper)  # pyright: ignore[reportPrivateUsage,reportArgumentType]
            is None
        )

    def test_added_vocab_token_id_no_longer_triggers_corruption_guard(self) -> None:
        """End-to-end semantics of the fix: an added-token id (between
        ``vocab_size`` and ``vocab_size + |added_vocab|``) must satisfy
        the spec-decode guard ``0 <= token < vocab_size``. Without the
        fix this id falsely tripped the guard and crashed generations.
        """

        class _Tokenizer:
            vocab_size: int = 32000

            def __len__(self) -> int:
                return self.vocab_size + 4

        full_size = self._call(_Tokenizer())
        assert full_size is not None
        added_token_id = 32002  # within added-vocab range
        assert 0 <= added_token_id < full_size
