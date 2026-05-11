"""Regression tests for ``GenerationStats.drafter_acceptance_fraction``.

Covers the Codex P2 finding (PR #19 round-(N+1)) that the property
mis-reported n-gram speculative runs as non-speculative because it
keyed on ``drafter_model_id`` being non-None, but n-gram speculation
intentionally runs without a drafter model id (in-process suffix
lookup, not a separate model).
"""

from __future__ import annotations

from exo.api.types.api import GenerationStats
from exo.shared.types.memory import Memory


def _stats(
    *,
    generation_tokens: int = 100,
    accepted: int = 25,
    drafter_model_id: str | None = None,
    draft_mode: str | None = None,
    num_draft_tokens: int | None = None,
) -> GenerationStats:
    return GenerationStats(
        prompt_tps=0.0,
        prompt_tokens=10,
        generation_tps=10.0,
        generation_tokens=generation_tokens,
        peak_memory_usage=Memory.from_bytes(0),
        accepted_draft_tokens=accepted,
        drafter_model_id=drafter_model_id,
        num_draft_tokens=num_draft_tokens,
        draft_mode=draft_mode,  # pyright: ignore[reportArgumentType]
    )


def test_acceptance_fraction_is_none_for_explicit_none_mode() -> None:
    """``draft_mode="none"`` is the canonical "no drafter" signal;
    acceptance fraction must be ``None`` regardless of any stale
    ``accepted_draft_tokens`` field on the payload."""
    stats = _stats(draft_mode="none", accepted=0)
    assert stats.drafter_acceptance_fraction is None


def test_acceptance_fraction_reports_for_ngram_runs() -> None:
    """Codex P2 (PR #19 round-(N+1)): n-gram speculation has no
    ``drafter_model_id`` because it's an in-process suffix lookup
    rather than a separate model. The acceptance fraction must
    still surface so n-gram A/B telemetry is meaningful.

    Pre-fix this returned ``None`` because ``drafter_model_id is
    None`` short-circuited the property."""
    stats = _stats(draft_mode="ngram", drafter_model_id=None, accepted=30)
    fraction = stats.drafter_acceptance_fraction
    assert fraction is not None, (
        "n-gram speculative runs must report an acceptance fraction "
        "even though they have no drafter_model_id"
    )
    assert abs(fraction - 0.30) < 1e-9


def test_acceptance_fraction_reports_for_model_runs() -> None:
    """Existing model-mode behaviour stays intact."""
    stats = _stats(
        draft_mode="model",
        drafter_model_id="some-org/drafter-7b",
        accepted=40,
    )
    fraction = stats.drafter_acceptance_fraction
    assert fraction is not None
    assert abs(fraction - 0.40) < 1e-9


def test_acceptance_fraction_legacy_payload_without_draft_mode() -> None:
    """Older recorded benches don't have ``draft_mode``; we must
    still honour the legacy heuristic (drafter_model_id present
    => speculative) so historical telemetry doesn't disappear."""
    legacy_with_drafter = _stats(
        draft_mode=None,
        drafter_model_id="legacy-org/drafter",
        accepted=10,
    )
    legacy_without_drafter = _stats(
        draft_mode=None,
        drafter_model_id=None,
        accepted=0,
    )
    assert legacy_with_drafter.drafter_acceptance_fraction is not None
    assert legacy_without_drafter.drafter_acceptance_fraction is None


def test_acceptance_fraction_zero_generation_tokens_returns_none() -> None:
    """Avoid divide-by-zero when no tokens were generated (e.g.
    immediate cancel or empty completion)."""
    stats = _stats(generation_tokens=0, accepted=0, draft_mode="ngram")
    assert stats.drafter_acceptance_fraction is None
