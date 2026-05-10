"""Validation tests for ``ChatCompletionRequest``.

These tests pin the API-level bounds on the speculative-decoding overrides
exposed via the OpenAI-compatible chat endpoint. The runner allocates a
fixed ``num_draft_tokens`` budget at warmup (``EXO_NUM_DRAFT_TOKENS``); a
per-request override above the budget would historically crash the runner
subprocess via an unhandled ``ValueError`` in ``PipelinedModelDrafter.__init__``
(regression: aborted K=8 sweep at 14:35:05 took the target rank with it,
leaving the drafter peer wedged in ``RunnerRunning`` while the respawned
target was stuck in ``RunnerIdle``).

The clamp inside ``generate.py`` defends the runner; the API bound here
exists only as a sanity guard against obviously-pathological values
(e.g. ``10**9``) so callers see a structured 422 instead of an opaque
mid-stream error. Codex flagged on PR #20 round 2 that an earlier
``= 32`` cap was too tight for benchmarking flows that sweep larger K
values when the operator has explicitly raised
``EXO_NUM_DRAFT_TOKENS``; the cap is now generous enough that the
runner's internal clamp is the authoritative bound for legitimate
sweeps.
"""

import pytest
from pydantic import ValidationError

from exo.api.types.api import (
    MAX_NUM_DRAFT_TOKENS_PER_REQUEST,
    ChatCompletionRequest,
)


def _minimal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def test_num_draft_tokens_default_is_none() -> None:
    request = ChatCompletionRequest.model_validate(_minimal_payload())
    assert request.num_draft_tokens is None


def test_num_draft_tokens_within_bounds_is_accepted() -> None:
    request = ChatCompletionRequest.model_validate(_minimal_payload(num_draft_tokens=4))
    assert request.num_draft_tokens == 4


def test_num_draft_tokens_at_upper_bound_is_accepted() -> None:
    request = ChatCompletionRequest.model_validate(
        _minimal_payload(num_draft_tokens=MAX_NUM_DRAFT_TOKENS_PER_REQUEST)
    )
    assert request.num_draft_tokens == MAX_NUM_DRAFT_TOKENS_PER_REQUEST


def test_num_draft_tokens_above_upper_bound_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(
            _minimal_payload(num_draft_tokens=MAX_NUM_DRAFT_TOKENS_PER_REQUEST + 1)
        )

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("num_draft_tokens",) and err["type"] == "less_than_equal"
        for err in errors
    )


def test_num_draft_tokens_benchmarking_sweep_value_is_accepted() -> None:
    """K=64 is a realistic benchmarking value when the operator has
    raised ``EXO_NUM_DRAFT_TOKENS``. Pre-fix the API hard-rejected
    anything above 32 with a 422 before the request could even reach
    the runner's clamp; post-fix the API only blocks pathological
    values, so legitimate K sweeps are no longer regressed (PR #20
    round 2 P2).
    """
    request = ChatCompletionRequest.model_validate(
        _minimal_payload(num_draft_tokens=64)
    )
    assert request.num_draft_tokens == 64


def test_num_draft_tokens_pathological_value_rejected() -> None:
    """The cap exists to reject genuinely malformed values like
    ``10**9``, which would otherwise reach the runner subprocess and
    trigger an OOM or unhandled ``ValueError`` in
    ``PipelinedModelDrafter.__init__``.
    """
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_payload(num_draft_tokens=10**9))


def test_num_draft_tokens_zero_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(_minimal_payload(num_draft_tokens=0))

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("num_draft_tokens",) and err["type"] == "greater_than_equal"
        for err in errors
    )


def test_num_draft_tokens_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_minimal_payload(num_draft_tokens=-3))
