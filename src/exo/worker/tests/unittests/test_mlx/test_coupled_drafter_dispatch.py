"""Dispatch-shape tests for :class:`CoupledModelDrafter`.

These tests exercise the Phase 2c integration seam between
:class:`exo.worker.engines.mlx.generator.coupled_drafter.CoupledModelDrafter`
and the :class:`Drafter`-protocol-shaped contract that
:func:`mlx_generate` consumes. They use a tiny in-memory Gemma 4 target
plus a stub drafter so the round loop runs end-to-end on CPU without
pulling the 78M-parameter gemma4_assistant weights into the test bus.

End-to-end parity (target-only vs MTP-accelerated produces byte-identical
tokens at temperature 0) lands as a separate manual / weight-loading
test in Phase 2d alongside the model-card placement work; here we
focus on the mechanics: the drafter satisfies the Drafter Protocol,
the prefill-capture-then-yield-bonus sequence emits the right
:class:`mlx_lm.GenerationResponse` shape, the metrics surface drives
``GenerationStats``, and the EOS / length / cancellation contracts
match the standard drafter path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, final

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.gemma4_text import ModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator.coupled_drafter import (
    CoupledModelDrafter,
    Gemma4MTPTargetAdapter,
)
from exo.worker.engines.mlx.generator.drafter import Drafter
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import CoupledDrafter
from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import attach_mtp_hooks

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


def _build_tiny_gemma4_with_hooks() -> Gemma4Model:
    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size=100,
        vocab_size_per_layer_input=100,
        sliding_window=32,
        sliding_window_pattern=2,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"],
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    model = Gemma4Model(args)
    model.eval()
    attach_mtp_hooks(model)
    return model


@final
class _StubGemma4Drafter(nn.Module):
    """Reused from :file:`test_coupled_drafter_round_loop.py` -- see that
    module for the full ``_mtp_rounds``-contract description. Returns
    drafts that always reject so the loop emits exactly one token per
    round (the target's bonus), keeping emission counts deterministic.
    """

    @final
    class _Config:
        block_size: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.config: _StubGemma4Drafter._Config = _StubGemma4Drafter._Config()
        self.accept_lens: list[int] = []
        self.bind_calls: int = 0
        self.set_shared_kv_calls: int = 0
        self.draft_block_calls: int = 0

    def bind(self, target_model: object) -> "_StubGemma4Drafter":
        del target_model
        self.bind_calls += 1
        return self

    def make_cache(self) -> list[Any]:
        return []

    def reset(self, target_model: object) -> list[Any]:
        self.bind(target_model)
        self.accept_lens = []
        return []

    def set_shared_kv(
        self,
        shared_kv_states: dict[str, tuple[mx.array, mx.array]],
        kv_offset: int | mx.array,
        position: int | mx.array | None = None,
        left_padding: mx.array | None = None,
    ) -> None:
        del shared_kv_states, kv_offset, position, left_padding
        self.set_shared_kv_calls += 1

    def draft_block(
        self,
        last_bonus: int,
        hidden: mx.array,
        cache: object,
        block_size: int,
        sampler: object,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        del last_bonus, hidden, cache, sampler
        self.draft_block_calls += 1
        return mx.zeros((1, block_size - 1), dtype=token_dtype)


@final
class _StubDetokenizer:
    """Minimal detokenizer surface consumed by :class:`CoupledModelDrafter`.

    The drafter calls only ``reset()``, ``add_token(int)``, ``finalize()``,
    and reads ``last_segment``. Any closer fidelity to the production
    :mod:`mlx_lm.tokenizer_utils` would couple these tests to that
    module's evolving contract; the stub is the smallest surface that
    satisfies the call sequence.
    """

    def __init__(self) -> None:
        self.last_segment: str = ""
        self.tokens: list[int] = []
        self.finalized: bool = False

    def reset(self) -> None:
        self.tokens = []
        self.last_segment = ""
        self.finalized = False

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        self.last_segment = f" t{token}"

    def finalize(self) -> None:
        self.finalized = True
        self.last_segment = ""


@final
class _StubTokenizer:
    """Minimal :class:`TokenizerWrapper`-shaped tokenizer for the drafter."""

    def __init__(self, eos_token_ids: list[int] | None = None) -> None:
        self.detokenizer: _StubDetokenizer = _StubDetokenizer()
        self.eos_token_ids: list[int] = list(eos_token_ids or [])


def _greedy_sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.int32)


# --------------------------------------------------------------------------- #
# Drafter-protocol conformance
# --------------------------------------------------------------------------- #


def test_coupled_drafter_satisfies_drafter_protocol() -> None:
    """The dispatch in ``mlx_generate`` types ``drafter: Drafter`` and
    relies on the runtime-checkable Protocol; the structural mismatch
    that would slip past a static type check (e.g. ``mode`` returning
    a non-DraftMode literal, ``stream`` missing an arg) must surface
    here, not at the first request."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)
    drafter = CoupledModelDrafter(
        target_adapter=adapter,
        drafter=_StubGemma4Drafter(),
        kind="mtp",
        num_draft_tokens=2,
    )

    assert isinstance(drafter, Drafter)
    assert drafter.mode == "model"
    assert drafter.kind == "mtp"
    assert drafter.num_draft_tokens == 2


def test_coupled_drafter_rejects_zero_k() -> None:
    """``num_draft_tokens=0`` is meaningless (no proposals = no
    speculation); the constructor must fail loudly so a misconfigured
    runner doesn't silently emit only bonus tokens."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        CoupledModelDrafter(
            target_adapter=adapter,
            drafter=_StubGemma4Drafter(),
            kind="mtp",
            num_draft_tokens=0,
        )


# --------------------------------------------------------------------------- #
# Stream behaviour
# --------------------------------------------------------------------------- #


def _run_stream(
    *,
    target: Gemma4Model,
    drafter: _StubGemma4Drafter,
    prompt_tokens: list[int],
    max_tokens: int,
    eos_token_ids: list[int] | None = None,
    sampler: Callable[[mx.array], mx.array] | None = None,
) -> tuple[list[GenerationResponse], _StubTokenizer]:
    """Drive ``CoupledModelDrafter.stream`` to completion and collect responses.

    Mirrors the call shape :func:`mlx_generate` uses: the drafter
    receives the prefill-tail (last 2 prompt tokens), a freshly-built
    cache covering the rest of the prompt, and the standard sampler /
    logits_processors / context_tokens triple.
    """
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", drafter),
        kind="mtp",
        num_draft_tokens=2,
    )
    tokenizer = _StubTokenizer(eos_token_ids)
    sampler_fn = sampler or _greedy_sampler

    prefill_prompt = prompt_tokens[:-2]
    decode_prompt = prompt_tokens[-2:]

    cache: list[Any] = cast("list[Any]", target.make_cache())
    if prefill_prompt:
        # ``target`` returns ``mx.array``-typed logits at runtime but the
        # callable surface is structurally generic; we discard the result
        # explicitly so basedpyright doesn't flag the unused expression.
        _ = target(mx.array([prefill_prompt]), cache=cache)

    # ``model`` is typed ``Model`` (a Protocol) on the production
    # signature; the runtime gemma4_text.Model satisfies it but the
    # static surface won't accept the concrete class without help.
    # ``tokenizer`` likewise: the production signature is
    # :class:`TokenizerWrapper` and our stub is structurally compatible
    # with the slots the drafter actually reaches.
    responses: list[GenerationResponse] = list(
        coupled.stream(
            model=cast("Model", cast("object", target)),
            tokenizer=cast("TokenizerWrapper", cast("object", tokenizer)),
            prompt=mx.array(decode_prompt),
            context_tokens=prompt_tokens,
            prompt_cache=cast("KVCacheType", cache),
            max_tokens=max_tokens,
            sampler=sampler_fn,
            logits_processors=[],
            prefill_step_size=1,
        )
    )
    return responses, tokenizer


def test_stream_yields_first_bonus_with_finish_reason_none() -> None:
    """The first emitted response carries the sampled bonus, real
    logprobs (we computed them ourselves before entering the round
    loop), and ``finish_reason=None`` so the caller's stop-sequence
    detection can run before the closing chunk fires."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=4,
    )

    assert len(responses) >= 2, "stream must yield at least the bonus + closing"
    first = responses[0]
    assert first.token != 0 or first.token == 0  # token is whatever sampler picked
    assert first.from_draft is False, "first bonus is not a drafted token"
    assert first.finish_reason is None
    assert first.generation_tokens == 1


def test_stream_does_not_flag_round_loop_tokens_as_from_draft() -> None:
    """Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569): every
    ``_mtp_rounds`` round emits ``accept_lens[i] + 1`` tokens (the
    accepted drafts plus one verifier bonus), but the coupled path
    receives them as a flat token stream without per-token
    provenance. Pre-fix every round-loop emission was tagged
    ``from_draft=True``, which let ``from_draft_count`` exceed
    ``proposed_draft_tokens`` on high-acceptance runs and corrupted
    acceptance-rate dashboards.

    The corrected contract: round-loop emissions carry
    ``from_draft=False`` (because we cannot honestly attribute each
    token), and the authoritative acceptance count is surfaced via
    :meth:`CoupledModelDrafter.metrics` (sum of
    ``drafter.accept_lens``). ``mlx_generate`` prefers the metric
    over the per-emit tally, so acceptance ratios stay bounded in
    ``[0, 1]``.
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=4,
    )

    assert all(not r.from_draft for r in responses), (
        "no coupled emission should claim from_draft attribution; "
        "the round-loop yields a flat stream of accepted-drafts + "
        "verifier-bonus mixed without per-token provenance, so the "
        "authoritative acceptance count comes from drafter.metrics()"
    )


def test_stream_respects_max_tokens() -> None:
    """``max_tokens`` is the upper bound on emitted tokens, including
    the first bonus. The caller's ``length`` finish reason fires when
    the budget runs out."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=3,
    )

    # The closing chunk is the last response; ``generation_tokens``
    # on it is the canonical emit count.
    closing = responses[-1]
    assert closing.generation_tokens <= 3
    assert closing.finish_reason in {"stop", "length"}


def test_stream_emits_eos_with_stop_finish_reason() -> None:
    """When the round loop yields an EOS token, the drafter must
    short-circuit emission and surface ``finish_reason="stop"`` --
    matching what mlx_lm's stream_generate does for non-spec runs."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    # Build a sampler that picks token 7 (our EOS) every time. This
    # makes the FIRST BONUS land on EOS, exercising the early-exit
    # path; the round loop never runs in this case.
    def _eos_sampler(logits: mx.array) -> mx.array:
        return mx.full(logits.shape[:-1], 7, dtype=mx.int32)

    responses, tokenizer = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=8,
        eos_token_ids=[7],
        sampler=_eos_sampler,
    )

    closing = responses[-1]
    assert closing.finish_reason == "stop"
    assert closing.token == 7
    assert tokenizer.detokenizer.finalized, "detokenizer must be finalised on close"


# --------------------------------------------------------------------------- #
# Metrics + telemetry
# --------------------------------------------------------------------------- #


def test_metrics_returns_zeros_before_stream_runs() -> None:
    """Pre-stream metrics are all zero -- ``GenerationStats``
    construction in :func:`mlx_generate` reads metrics() at finish
    time, so this case shouldn't fire in production, but exposing
    zeroes for unrun streams keeps the contract sensible."""
    target = _build_tiny_gemma4_with_hooks()
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", _StubGemma4Drafter()),
        kind="mtp",
        num_draft_tokens=2,
    )

    metrics = coupled.metrics()
    assert metrics["spec_decode_rounds"] == 0
    assert metrics["proposed_draft_tokens"] == 0
    assert metrics["accepted_draft_tokens"] == 0


def test_metrics_accepted_never_exceeds_proposed() -> None:
    """Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569): the
    acceptance ratio (``accepted / proposed``) MUST stay bounded in
    ``[0, 1]``. Pre-fix every round-loop emit was tagged
    ``from_draft=True`` and the per-round verifier bonus was counted
    as a draft, so a full-acceptance round of K drafts produced K+1
    "accepted" tokens against K proposed, inflating the ratio above
    1.0 and corrupting acceptance dashboards.

    The corrected accounting sources ``accepted_draft_tokens`` from
    ``sum(drafter.accept_lens)`` (the canonical mlx-vlm tally of
    actual drafts the verifier accepted), which is bounded by
    ``rounds * (block_size - 1) == proposed_draft_tokens`` because
    each round can accept at most ``block_size - 1`` drafts.
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", drafter),
        kind="mtp",
        num_draft_tokens=2,
    )
    tokenizer = _StubTokenizer()

    cache: list[Any] = cast("list[Any]", target.make_cache())
    _ = target(mx.array([[1, 2]]), cache=cache)
    _ = list(
        coupled.stream(
            model=cast("Model", cast("object", target)),
            tokenizer=cast("TokenizerWrapper", cast("object", tokenizer)),
            prompt=mx.array([3, 4]),
            context_tokens=[1, 2, 3, 4],
            prompt_cache=cast("KVCacheType", cache),
            max_tokens=8,
            sampler=_greedy_sampler,
            logits_processors=[],
        )
    )

    metrics = coupled.metrics()
    proposed = metrics["proposed_draft_tokens"]
    accepted = metrics["accepted_draft_tokens"]
    assert accepted >= 0, f"accepted_draft_tokens must be non-negative; got {accepted}"
    assert accepted <= proposed, (
        f"accepted_draft_tokens ({accepted}) must not exceed "
        f"proposed_draft_tokens ({proposed}) -- pre-fix the verifier "
        f"bonus was double-counted into the acceptance tally"
    )


def test_stream_prompt_tps_brackets_actual_prefill_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2 (PR #25 round-(N+0), coupled_drafter.py:484): pre-fix, the
    prompt-TPS timer was started AFTER the prefill ``self._target_adapter(...)``
    call had already returned, so ``prompt_time`` was effectively
    zero and ``prompt_tps`` came out as a meaningless huge number
    (or zero, when ``time.perf_counter()`` returned the same float
    twice in a row). Downstream telemetry treated this as a real
    measurement -- especially via the
    ``prefill_tps`` fallback to ``out.prompt_tps`` -- and broke
    coupled-vs-standard performance comparisons.

    Pin the corrected behaviour: the prefill call must run INSIDE
    the timed window. We monkeypatch ``time.perf_counter`` with a
    deterministic clock that advances by a known amount across the
    prefill call, then assert ``prompt_tps`` matches
    ``prompt_tail_size / prefill_seconds`` -- i.e. the measurement
    actually reflects the prefill cost.
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    timeline = iter([0.0, 0.5, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0])
    fallback = [4.0]

    def _fake_perf_counter() -> float:
        try:
            return next(timeline)
        except StopIteration:
            fallback[0] += 0.001
            return fallback[0]

    import exo.worker.engines.mlx.generator.coupled_drafter as module_under_test

    monkeypatch.setattr(module_under_test.time, "perf_counter", _fake_perf_counter)

    responses, _ = _run_stream(
        target=target,
        drafter=drafter,
        prompt_tokens=[1, 2, 3, 4],
        max_tokens=4,
    )

    first = responses[0]
    # prompt size = 2 (prefill-tail [3, 4]); the fake clock advanced by
    # 0.5s across the prefill call (0.0 -> 0.5), so prompt_tps must be
    # 2 / 0.5 = 4.0 tokens/second. Pre-fix this would have been 0.0
    # (zero elapsed) or some huge garbage value depending on when
    # ``perf_counter`` was sampled.
    assert first.prompt_tokens == 2
    assert abs(first.prompt_tps - 4.0) < 1e-6, (
        f"prompt_tps must reflect prefill cost (expected 4.0 from "
        f"prompt_tail=2 / prefill_dt=0.5s), got {first.prompt_tps}"
    )


def test_metrics_after_stream_reflects_round_count() -> None:
    """Each entry in ``drafter.accept_lens`` is a completed round; the
    drafter appends to it from inside ``_mtp_rounds``. After a stream
    that emits ``max_tokens`` total, the round count must be at least 1
    (the loop ran) and ``proposed_draft_tokens`` must scale with the
    round count and the configured block size."""
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()
    coupled = CoupledModelDrafter(
        target_adapter=Gemma4MTPTargetAdapter(target),
        drafter=cast("nn.Module", drafter),
        kind="mtp",
        num_draft_tokens=2,
    )
    tokenizer = _StubTokenizer()

    cache: list[Any] = cast("list[Any]", target.make_cache())
    # Prefill prompt[:-2] outside the drafter, mirroring mlx_generate.
    _ = target(mx.array([[1, 2]]), cache=cache)

    _ = list(
        coupled.stream(
            model=cast("Model", cast("object", target)),
            tokenizer=cast("TokenizerWrapper", cast("object", tokenizer)),
            prompt=mx.array([3, 4]),
            context_tokens=[1, 2, 3, 4],
            prompt_cache=cast("KVCacheType", cache),
            max_tokens=4,
            sampler=_greedy_sampler,
            logits_processors=[],
        )
    )

    metrics = coupled.metrics()
    assert metrics["spec_decode_rounds"] >= 1, (
        "round loop must have run at least once for max_tokens=4"
    )
    # block_size=4 → 3 drafts proposed per round.
    assert metrics["proposed_draft_tokens"] == metrics["spec_decode_rounds"] * 3


# --------------------------------------------------------------------------- #
# Coupled telemetry gating (Codex P2 PR #25 round-(N+1))
# --------------------------------------------------------------------------- #


class TestResolveCoupledDrafterTelemetry:
    """Pin the contract of :func:`_resolve_coupled_drafter_telemetry`.

    Codex P2 (PR #25 round-(N+1), generate.py:1710): the telemetry
    block in :func:`mlx_generate` previously gated coupled-drafter
    fields on the RESOURCE signal (``coupled_drafter_active`` --
    "we loaded a coupled drafter and the request resolved to
    ``draft_mode='model'``"). The helper extracted in this commit
    gates on the DISPATCH signal (``coupled_dispatch_fired``) so a
    loaded-but-not-dispatched coupled drafter never leaks
    ``drafter_model_id`` / ``drafter_kind`` / ``num_draft_tokens``
    onto a request that actually ran with ``draft_mode='none'``.

    All currently dispatched kinds (``"mtp"``, ``"dflash"``) drive
    speculation end-to-end, so the dispatch signal SHOULD fire for
    them. The helper's ``coupled_dispatch_fired=False`` branch
    remains the canonical fallback for any future kind that lands
    on the loader before its dispatch wiring catches up, or any
    runtime fallback (e.g. an attach-hook ``TypeError``) that
    forces :func:`make_drafter` to take over.
    """

    @staticmethod
    def _make_coupled_drafter(kind: str) -> CoupledDrafter:
        from exo.shared.types.common import ModelId
        from exo.worker.engines.mlx.utils_mlx import CoupledDrafterKind

        return CoupledDrafter(
            model_id=ModelId("mlx-community/coupled-test-drafter"),
            kind=cast("CoupledDrafterKind", kind),
            model=object(),
        )

    def test_dispatch_fired_returns_telemetry(self) -> None:
        from exo.worker.engines.mlx.generator.generate import (
            _resolve_coupled_drafter_telemetry,  # pyright: ignore[reportPrivateUsage]
        )

        coupled = self._make_coupled_drafter("mtp")
        drafter_id, drafter_kind, num_draft_tokens = _resolve_coupled_drafter_telemetry(
            coupled_dispatch_fired=True,
            coupled_drafter=coupled,
            effective_num_draft_tokens=4,
        )
        assert drafter_id == "mlx-community/coupled-test-drafter"
        assert drafter_kind == "mtp"
        assert num_draft_tokens == 4

    def test_dispatch_not_fired_zeros_telemetry_even_with_loaded_drafter(
        self,
    ) -> None:
        """The fallback path: ``coupled_drafter`` is loaded but
        dispatch chose ``make_drafter(mode='none')`` (e.g. an
        attach-hook ``TypeError`` forced fallback, or the kind
        landed on the loader before its dispatch wiring caught
        up). Coupled telemetry must be zeroed so
        ``GenerationStats`` doesn't misattribute the request.

        We use ``"dflash"`` as the loaded kind here as a
        representative coupled kind; the helper itself doesn't
        gate on kind, only on the dispatch signal.
        """
        from exo.worker.engines.mlx.generator.generate import (
            _resolve_coupled_drafter_telemetry,  # pyright: ignore[reportPrivateUsage]
        )

        coupled = self._make_coupled_drafter("dflash")
        drafter_id, drafter_kind, num_draft_tokens = _resolve_coupled_drafter_telemetry(
            coupled_dispatch_fired=False,
            coupled_drafter=coupled,
            effective_num_draft_tokens=4,
        )
        assert drafter_id is None, (
            "coupled fallback (dispatch did not fire) must NOT stamp drafter_model_id"
        )
        assert drafter_kind is None, (
            "coupled fallback must NOT stamp drafter_kind -- pre-fix this "
            "leaked 'dflash' onto draft_mode='none' requests"
        )
        assert num_draft_tokens is None, (
            "coupled fallback must NOT stamp num_draft_tokens -- the "
            "fallback runs no speculation"
        )

    def test_no_coupled_drafter_loaded_zeros_telemetry(self) -> None:
        """Standard / pipelined / ngram / none requests don't carry a
        coupled drafter; helper returns the empty tuple so the
        caller falls through to its other branches.
        """
        from exo.worker.engines.mlx.generator.generate import (
            _resolve_coupled_drafter_telemetry,  # pyright: ignore[reportPrivateUsage]
        )

        result = _resolve_coupled_drafter_telemetry(
            coupled_dispatch_fired=False,
            coupled_drafter=None,
            effective_num_draft_tokens=4,
        )
        assert result == (None, None, None)

    def test_dispatch_fired_with_no_drafter_is_defensive_zero(self) -> None:
        """The dispatch signal can't be ``True`` while ``coupled_drafter
        is None`` in the production code path, but the helper still
        defends against it: returning the empty tuple is safer than
        constructing an ``str(None)`` model id.
        """
        from exo.worker.engines.mlx.generator.generate import (
            _resolve_coupled_drafter_telemetry,  # pyright: ignore[reportPrivateUsage]
        )

        result = _resolve_coupled_drafter_telemetry(
            coupled_dispatch_fired=True,
            coupled_drafter=None,
            effective_num_draft_tokens=4,
        )
        assert result == (None, None, None)


# --------------------------------------------------------------------------- #
# Logits processors flow through the coupled round loop
# (Codex P1 PR #25 round-(N+3))
# --------------------------------------------------------------------------- #


class TestProcessorAwareSampler:
    """Pin the contract of the wrapped sampler used in
    :class:`CoupledModelDrafter.stream`.

    Codex P1 (PR #25 round-(N+3), coupled_drafter.py:566): pre-fix
    only :func:`_select_first_bonus` applied per-request
    ``logits_processors`` (repetition / presence / frequency
    penalties, the bench EOS-ban processor); ``run_coupled_round_loop``
    received the bare ``sampler`` and so every token after the first
    bypassed those processors. Coupled requests therefore diverged
    from non-coupled decoding semantics from token 2 onwards.

    The fix wraps ``sampler`` so each ``sampler(logits)`` call inside
    mlx-vlm's ``_mtp_rounds`` first runs every processor against the
    running emitted-token history. These tests pin the wrapper
    behaviour without standing up the full round loop.
    """

    def test_empty_processors_returns_sampler_unchanged(self) -> None:
        from exo.worker.engines.mlx.generator.coupled_drafter import (
            _make_processor_aware_sampler,  # pyright: ignore[reportPrivateUsage]
        )

        running: list[int] = []

        def base_sampler(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        wrapped = _make_processor_aware_sampler(
            sampler=base_sampler,
            logits_processors=[],
            running_tokens=running,
        )
        assert wrapped is base_sampler, (
            "empty processor list must short-circuit to the original "
            "sampler so the no-processor path pays no per-call overhead"
        )

    def test_processor_runs_on_every_call_with_current_running_tokens(self) -> None:
        from exo.worker.engines.mlx.generator.coupled_drafter import (
            _make_processor_aware_sampler,  # pyright: ignore[reportPrivateUsage]
        )

        running: list[int] = [10, 20]
        captured_prev: list[list[int]] = []

        def proc(prev: mx.array, logits: mx.array) -> mx.array:
            captured_prev.append([int(t) for t in cast(list[int], prev.tolist())])
            return logits

        def base_sampler(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        wrapped = _make_processor_aware_sampler(
            sampler=base_sampler,
            logits_processors=[proc],
            running_tokens=running,
        )

        logits = mx.array([[0.1, 0.2, 0.3]])
        _ = wrapped(logits)
        running.append(30)
        _ = wrapped(logits)
        running.append(40)
        _ = wrapped(logits)

        assert captured_prev == [[10, 20], [10, 20, 30], [10, 20, 30, 40]], (
            "each wrapped sampler call must snapshot the LATEST "
            "running_tokens; pre-fix the processor never ran at all "
            "inside the round loop"
        )

    def test_multiple_processors_apply_in_order(self) -> None:
        from exo.worker.engines.mlx.generator.coupled_drafter import (
            _make_processor_aware_sampler,  # pyright: ignore[reportPrivateUsage]
        )

        running: list[int] = [1]
        marks: list[str] = []

        def proc_a(prev: mx.array, logits: mx.array) -> mx.array:
            del prev
            marks.append("a")
            return logits + 1.0

        def proc_b(prev: mx.array, logits: mx.array) -> mx.array:
            del prev
            marks.append("b")
            return logits * 2.0

        captured_logits: list[mx.array] = []

        def base_sampler(logits: mx.array) -> mx.array:
            captured_logits.append(logits)
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        wrapped = _make_processor_aware_sampler(
            sampler=base_sampler,
            logits_processors=[proc_a, proc_b],
            running_tokens=running,
        )
        _ = wrapped(mx.array([[0.0, 1.0, 2.0]]))

        assert marks == ["a", "b"], (
            "processors must apply in the order supplied (matching "
            "_select_first_bonus and the standard ModelDrafter path)"
        )
        # Final logits seen by the sampler: ((x + 1) * 2)
        # = ((0.0 + 1.0) * 2, (1.0 + 1.0) * 2, (2.0 + 1.0) * 2)
        # = (2.0, 4.0, 6.0)
        assert len(captured_logits) == 1
        final = captured_logits[0]
        expected = [2.0, 4.0, 6.0]
        actual = [float(x) for x in cast(list[float], final[0].tolist())]
        for got, want in zip(actual, expected, strict=True):
            assert abs(got - want) < 1e-6, (
                f"processor chain output mismatch; got {actual}, want {expected}"
            )
