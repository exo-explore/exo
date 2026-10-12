# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
"""MTP speculative decoding returns exactly the tokens of normal decoding.

Uses the tiny randomly initialized DeepSeek V4 with a random MTP block, so
drafts are mostly wrong; an oracle drafter covers the accepted paths.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.generate import GenerationBatch, SequenceStateMachine
from mlx_lm.models import deepseek_v4 as dv4
from mlx_lm.models.cache import BatchRotatingKVCache

from exo.worker.engines.mlx import deepseek_v4_mtp as mtp_module
from exo.worker.engines.mlx import deepseek_v4_speculative as speculative
from exo.worker.engines.mlx.patches import deepseek_v4_decode_kernels as kernels
from exo.worker.engines.mlx.patches import deepseek_v4_indexer as indexer_patch
from exo.worker.engines.mlx.patches import opt_batch_gen
from exo.worker.engines.mlx.tests.tiny_deepseek_v4 import (
    SLIDING_WINDOW,
    VOCAB_SIZE,
    tiny_model,
)

_PROMPT_LENGTH = 30
_STEPS = 40
# Speculative verification computes logits for several tokens at once, which
# rounds slightly differently from one-token steps, so greedy decoding of the
# random tiny model may legitimately flip where its top two logits nearly tie.
# Token sequences are compared up to the first such near-tie.
_NEAR_TIE = 1e-3
_MIN_COMPARED_STEPS = 30
# Relative logit error after rolling back: correct rollbacks stay below 2e-4,
# while leaving the compressor state unrolled gives 2e-2 or more.
_ROLLBACK_TOLERANCE = 1e-3


def _apply_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GenerationBatch, "_step", opt_batch_gen._patched_step)
    monkeypatch.setattr(GenerationBatch, "extend", speculative._flushing_extend)
    monkeypatch.setattr(GenerationBatch, "filter", speculative._flushing_filter)
    monkeypatch.setattr(
        dv4.DeepseekV4Cache, "accumulate_windows", speculative._recording_accumulate
    )
    monkeypatch.setattr(
        dv4.DeepseekV4Model, "__call__", mtp_module._capturing_inner_call
    )
    monkeypatch.setattr(dv4.HyperConnection, "hc_pre", kernels._patched_hc_pre)
    monkeypatch.setattr(dv4, "_hc_expand_ops", kernels._patched_hc_expand_ops)
    monkeypatch.setattr(dv4.V4Attention, "__call__", kernels._patched_attention)
    monkeypatch.setattr(dv4.Indexer, "__call__", indexer_patch._patched_call)


@pytest.fixture(autouse=True)
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_patches(monkeypatch)


def _prompt() -> list[int]:
    rng = np.random.default_rng(3)
    return [int(t) for t in rng.integers(0, VOCAB_SIZE, _PROMPT_LENGTH)]


def _greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _batch(model: dv4.Model, prompt: list[int]) -> GenerationBatch:
    cache = model.make_cache()
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    return GenerationBatch(
        model=model,
        uids=[0],
        inputs=mx.array([prompt[-1]]),
        prompt_cache=cache,
        tokens=[list(prompt)],
        samplers=cast(list[Callable[[mx.array], mx.array]], [None]),
        fallback_sampler=_greedy,
        logits_processors=[[]],
        state_machines=[SequenceStateMachine()],
        max_tokens=[10_000],
    )


def _generate(
    model: dv4.Model,
    after_step: Callable[[GenerationBatch, int], None] | None = None,
) -> tuple[list[int], list[float], GenerationBatch]:
    """Generated tokens and the gap between the top two logprobs at each."""
    batch = _batch(model, _prompt())
    tokens: list[int] = []
    gaps: list[float] = []
    for step in range(_STEPS):
        response = batch.next()[0]
        tokens.append(response.token)
        top_two = mx.sort(response.logprobs.astype(mx.float32))[-2:]
        gaps.append((top_two[1] - top_two[0]).item())
        if after_step is not None:
            after_step(batch, step)
    return tokens, gaps, batch


@dataclass(frozen=True)
class _Reference:
    tokens: list[int]
    compared: int


def _attach_random_mtp(model: dv4.Model) -> None:
    mtp = mtp_module.DeepseekV4MTP(model.args)
    mx.random.seed(11)
    params = [
        (
            path,
            (mx.random.normal(value.shape) * 0.05).astype(value.dtype)
            if mx.issubdtype(value.dtype, mx.floating)
            else value,
        )
        for path, value in cast(
            list[tuple[str, mx.array]], tree_flatten(mtp.parameters())
        )
    ]
    mtp.update(tree_unflatten(params))
    object.__setattr__(mtp, "exo_group", None)
    object.__setattr__(model, "exo_mtp", mtp)


@pytest.fixture(scope="module")
def reference() -> _Reference:
    model = tiny_model()
    with pytest.MonkeyPatch.context() as monkeypatch:
        _apply_patches(monkeypatch)
        tokens, gaps, batch = _generate(model)
    assert speculative._state(batch) is None
    compared = next((i for i, gap in enumerate(gaps) if gap < _NEAR_TIE), _STEPS)
    assert compared >= _MIN_COMPARED_STEPS
    return _Reference(tokens, compared)


def test_random_drafts_match_normal_decoding(reference: _Reference) -> None:
    model = tiny_model()
    _attach_random_mtp(model)
    tokens, _, batch = _generate(model)
    assert tokens[: reference.compared] == reference.tokens[: reference.compared]
    state = speculative._state(batch)
    assert state is not None and state.rounds > 0


def test_accepted_drafts_match_normal_decoding(
    reference: _Reference, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drafts that are right except at every third call, so rounds accept
    zero, one or all drafts and roll the caches back by different amounts."""
    model = tiny_model()
    _attach_random_mtp(model)
    original_draft = speculative._draft
    calls = 0

    def oracle_draft(
        model: dv4.Model,
        mtp: mtp_module.DeepseekV4MTP,
        state: speculative._SpeculativeState,
        hidden: mx.array,
        next_ids: mx.array,
    ) -> mx.array:
        nonlocal calls
        original_draft(model, mtp, state, hidden, next_ids)
        calls += 1
        # Generated tokens the MTP has seen so far.
        seen = state.mtp_cache.local.offset
        drafts: list[int] = []
        for j in range(speculative.DRAFT_TOKENS):
            truth = reference.tokens
            token = truth[seen + j] if seen + j < len(truth) else 0
            if (calls + j) % 3 == 0:
                token = (token + 1) % VOCAB_SIZE
            drafts.append(token)
        return mx.array(drafts, dtype=mx.uint32)

    monkeypatch.setattr(speculative, "_draft", oracle_draft)
    tokens, _, batch = _generate(model)
    assert tokens[: reference.compared] == reference.tokens[: reference.compared]
    state = speculative._state(batch)
    assert state is not None
    assert state.tokens / state.rounds > 1.5
    assert all(hits > 0 for hits in state.accepted_per_position)


def test_flush_mid_round_matches_normal_decoding(reference: _Reference) -> None:
    """Leaving speculative mode with a verification in flight or accepted
    drafts not yet returned (as a batch change does) loses nothing."""
    model = tiny_model()
    _attach_random_mtp(model)
    flushes = 0

    def flush_sometimes(batch: GenerationBatch, step: int) -> None:
        nonlocal flushes
        state = speculative._state(batch)
        if state is None or step % 3 != 0:
            return
        if state.pending is not None or state.inflight is not None:
            speculative.flush(batch)
            flushes += 1

    tokens, _, _ = _generate(model, flush_sometimes)
    assert tokens[: reference.compared] == reference.tokens[: reference.compared]
    assert flushes > 0


def test_batch_caches_disable_speculation() -> None:
    """After a batch of several requests shrinks back to one, its caches are
    still batch caches; speculation must leave them alone."""
    model = tiny_model()
    _attach_random_mtp(model)
    batch = _batch(model, _prompt())
    assert speculative._usable(batch) is not None
    first = cast(dv4.DeepseekV4Cache, batch.prompt_cache[0])
    first.local = BatchRotatingKVCache(SLIDING_WINDOW, left_padding=[0])  # pyright: ignore[reportAttributeAccessIssue]
    assert speculative._usable(batch) is None


@pytest.mark.parametrize("prompt_length", [14, 15, 16, 17, 30, 31, 126, 127, 128])
@pytest.mark.parametrize("keep", [1, 2])
def test_keep_tokens_equals_feeding_only_the_kept_tokens(
    prompt_length: int, keep: int
) -> None:
    """Rolling a 3-token step back to its first ``keep`` tokens leaves the
    caches as if only those tokens had been fed (window and compressor
    boundaries fall inside the step for some prompt lengths)."""
    model = tiny_model()
    rng = np.random.default_rng(prompt_length)
    tokens = mx.array(rng.integers(0, VOCAB_SIZE, (1, prompt_length + 4)))
    step = tokens[:, prompt_length : prompt_length + 3]
    following = tokens[:, prompt_length + 3 :]

    expected_cache = model.make_cache()
    mx.eval(model(tokens[:, :prompt_length], cache=expected_cache))
    mx.eval(model(step[:, :keep], cache=expected_cache))
    expected = model(following, cache=expected_cache)

    cache = model.make_cache()
    mx.eval(model(tokens[:, :prompt_length], cache=cache))
    snapshots = [speculative._snapshot(cast(dv4.DeepseekV4Cache, c)) for c in cache]
    speculative._recording = {}
    try:
        mx.eval(model(step, cache=cache))
        recorded = speculative._recording
    finally:
        speculative._recording = None
    for c, snapshot in zip(cache, snapshots, strict=True):
        speculative._keep_tokens(
            cast(dv4.DeepseekV4Cache, c), snapshot, recorded, keep=keep, total=3
        )
    actual = model(following, cache=cache)

    assert _relative_error(expected, actual) < _ROLLBACK_TOLERANCE


def _relative_error(reference: mx.array, actual: mx.array) -> float:
    reference = reference.astype(mx.float32)
    actual = actual.astype(mx.float32)
    error = mx.sqrt(mx.sum(mx.square(reference - actual)))
    return (error / mx.sqrt(mx.sum(mx.square(reference)))).item()
