"""Unit tests for ``chat_request_to_text_generation`` request forwarding.

These tests pin down which fields of :class:`ChatCompletionRequest`
are forwarded onto :class:`TextGenerationTaskParams`. The
speculative-decoding overrides (``use_drafter``, ``num_draft_tokens``,
``draft_mode``) are exo extensions to the OpenAI schema -- if any of
them silently drops between the API surface and the runner, callers
get the runner's process defaults instead of the requested per-request
behaviour, which (a) makes A/B experiments invisible and (b) breaks
the n-gram / "none" override paths the runner now supports via
:func:`resolve_draft_mode`.
"""

from __future__ import annotations

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.types.common import ModelId


def _request(**overrides: object) -> ChatCompletionRequest:
    """Minimal ``ChatCompletionRequest`` plus per-test overrides."""
    base: dict[str, object] = {
        "model": ModelId("mlx-community/test-model"),
        "messages": [ChatCompletionMessage(role="user", content="hello")],
    }
    base.update(overrides)
    return ChatCompletionRequest.model_validate(base)


@pytest.mark.asyncio
async def test_forwards_use_drafter() -> None:
    params = await chat_request_to_text_generation(_request(use_drafter=False))
    assert params.use_drafter is False


@pytest.mark.asyncio
async def test_forwards_num_draft_tokens() -> None:
    params = await chat_request_to_text_generation(_request(num_draft_tokens=12))
    assert params.num_draft_tokens == 12


@pytest.mark.asyncio
async def test_forwards_draft_mode_model() -> None:
    # Explicit "model" mode must round-trip to the runner so the
    # external-drafter loop is selected even when the runner's process
    # default is "ngram" or "none".
    params = await chat_request_to_text_generation(_request(draft_mode="model"))
    assert params.draft_mode == "model"


@pytest.mark.asyncio
async def test_forwards_draft_mode_ngram() -> None:
    # The n-gram path was the originally-flagged regression: callers
    # cannot opt into in-context lookahead per request without this
    # forwarding.
    params = await chat_request_to_text_generation(_request(draft_mode="ngram"))
    assert params.draft_mode == "ngram"


@pytest.mark.asyncio
async def test_forwards_draft_mode_none() -> None:
    # "none" must also forward so callers can force non-speculative
    # behaviour for a single benchmark run while leaving the runner
    # default intact for everyone else.
    params = await chat_request_to_text_generation(_request(draft_mode="none"))
    assert params.draft_mode == "none"


@pytest.mark.asyncio
async def test_unset_draft_mode_stays_none() -> None:
    # Callers that omit ``draft_mode`` get whatever the runner's
    # process default resolves to. The adapter must not synthesize a
    # value here -- it has to be ``None`` so
    # :func:`resolve_draft_mode` falls back to the env / config default.
    params = await chat_request_to_text_generation(_request())
    assert params.draft_mode is None


@pytest.mark.asyncio
async def test_explicit_draft_mode_does_not_disturb_use_drafter() -> None:
    # ``use_drafter`` and ``draft_mode`` are independently forwarded
    # so the runner's resolution helper sees both signals; previously
    # only ``use_drafter`` made it through, which collapsed the
    # caller's intent down to a single boolean.
    params = await chat_request_to_text_generation(
        _request(use_drafter=True, draft_mode="ngram", num_draft_tokens=4)
    )
    assert params.use_drafter is True
    assert params.draft_mode == "ngram"
    assert params.num_draft_tokens == 4
