"""Regression tests for speculative-decode generation telemetry."""

from __future__ import annotations

from exo.api.types.api import GenerationStats
from exo.shared.types.memory import Memory


def _stats(
    *,
    generation_tokens: int = 100,
    accepted: int = 25,
    proposed: int = 0,
    drafter_model_id: str | None = None,
    draft_mode: str | None = None,
    drafter_kind: str | None = None,
    num_draft_tokens: int | None = None,
) -> GenerationStats:
    return GenerationStats(
        prompt_tps=0.0,
        prompt_tokens=10,
        generation_tps=10.0,
        generation_tokens=generation_tokens,
        peak_memory_usage=Memory.from_bytes(0),
        accepted_draft_tokens=accepted,
        proposed_draft_tokens=proposed,
        drafter_model_id=drafter_model_id,
        num_draft_tokens=num_draft_tokens,
        draft_mode=draft_mode,  # pyright: ignore[reportArgumentType]
        drafter_kind=drafter_kind,  # pyright: ignore[reportArgumentType]
    )


def test_acceptance_fraction_is_none_for_explicit_none_mode() -> None:
    stats = _stats(draft_mode="none", accepted=0)
    assert stats.drafter_acceptance_fraction is None


def test_acceptance_fraction_reports_for_model_runs() -> None:
    stats = _stats(
        draft_mode="model",
        drafter_model_id="some-org/drafter-7b",
        accepted=40,
    )
    assert stats.drafter_acceptance_fraction == 0.40


def test_acceptance_metrics_report_for_native_mtp_runs_without_drafter_id() -> None:
    stats = _stats(
        draft_mode="model",
        drafter_model_id=None,
        drafter_kind="native_mtp",
        num_draft_tokens=2,
        accepted=25,
        proposed=40,
    )
    assert stats.drafter_acceptance_fraction == 0.25
    assert stats.drafter_acceptance_rate == 0.625


def test_acceptance_fraction_legacy_payload_without_draft_mode() -> None:
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
    stats = _stats(generation_tokens=0, accepted=0, draft_mode="model")
    assert stats.drafter_acceptance_fraction is None
