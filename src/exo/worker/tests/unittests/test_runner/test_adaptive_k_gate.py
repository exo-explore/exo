"""Unit tests for the adaptive-K acceptance gate.

``_acceptance_fraction_for_adaptive_k`` decides which generation
responses contribute observations to the rolling drafter-acceptance
window that drives :func:`adaptive_num_draft_tokens`. The rolling
window directly steers the next request's ``num_draft_tokens``, so
the gate's correctness matters: a misgated sample either poisons
the controller (a non-spec request contributing 0/N) or starves
it (a real spec round silently dropped).

The previously-flagged regression was n-gram speculation: the
n-gram strategy sets ``draft_mode="ngram"`` with no drafter model
id (it speculates from the in-context suffix without loading a
separate model), and the old gate keyed off
``drafter_model_id is not None`` so every n-gram round was silently
dropped under ``EXO_DRAFT_MODE=ngram`` + ``EXO_ADAPTIVE_DRAFT_TOKENS=1``,
pinning K at the fallback value forever. The new gate keys off
``draft_mode`` directly, which is populated for both ``model`` and
``ngram`` runs.
"""

from __future__ import annotations

import math

from exo.api.types.api import GenerationStats
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.runner.llm_inference.batch_generator import (
    _acceptance_fraction_for_adaptive_k,  # pyright: ignore[reportPrivateUsage]
)


def _stats(
    *,
    draft_mode: str | None,
    generation_tokens: int,
    accepted_draft_tokens: int = 0,
    drafter_model_id: str | None = None,
) -> GenerationStats:
    """Build a ``GenerationStats`` with the fields the gate inspects.

    The other fields (TPS, prompt-token counts, peak memory) are
    irrelevant to this gate; we set them to plausible values so
    Pydantic strict-mode validation succeeds.
    """
    return GenerationStats(
        prompt_tps=1.0,
        generation_tps=1.0,
        prompt_tokens=1,
        generation_tokens=generation_tokens,
        peak_memory_usage=Memory.from_gb(1.0),
        drafter_model_id=drafter_model_id,
        accepted_draft_tokens=accepted_draft_tokens,
        draft_mode=draft_mode,  # pyright: ignore[reportArgumentType]
    )


def _response(stats: GenerationStats | None) -> GenerationResponse:
    return GenerationResponse(
        text="",
        token=0,
        stats=stats,
        usage=None,
    )


class TestAcceptanceFractionForAdaptiveK:
    def test_model_mode_records_accepted_fraction(self) -> None:
        # External-drafter run with a quarter of generated tokens accepted.
        stats = _stats(
            draft_mode="model",
            generation_tokens=8,
            accepted_draft_tokens=2,
            drafter_model_id="mlx-community/test-drafter",
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 0.25)

    def test_ngram_mode_records_accepted_fraction_without_drafter_model_id(
        self,
    ) -> None:
        # n-gram speculation has no drafter model; the new gate must
        # still record this sample so adaptive K converges under
        # ``EXO_DRAFT_MODE=ngram``. This is the previously-dropped path.
        stats = _stats(
            draft_mode="ngram",
            generation_tokens=10,
            accepted_draft_tokens=4,
            drafter_model_id=None,
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 0.4)

    def test_none_mode_skips_record(self) -> None:
        # Non-speculative requests carry no drafter signal; recording
        # them would dilute the rolling window with zeroes.
        stats = _stats(
            draft_mode="none",
            generation_tokens=5,
            accepted_draft_tokens=0,
        )
        assert _acceptance_fraction_for_adaptive_k(_response(stats)) is None

    def test_unknown_mode_skips_record(self) -> None:
        # Defensive: if a future code path emits a stats payload with
        # ``draft_mode=None`` (e.g. image gen extending the same
        # response shape), the gate refuses to record rather than
        # poisoning the controller.
        stats = _stats(
            draft_mode=None,
            generation_tokens=5,
            accepted_draft_tokens=0,
        )
        assert _acceptance_fraction_for_adaptive_k(_response(stats)) is None

    def test_empty_generation_skips_record(self) -> None:
        # Immediate stop-sequence hit on prefill produces zero
        # generated tokens. There's no acceptance signal in that
        # request, and the division would raise ``ZeroDivisionError``.
        stats = _stats(
            draft_mode="model",
            generation_tokens=0,
            accepted_draft_tokens=0,
            drafter_model_id="mlx-community/test-drafter",
        )
        assert _acceptance_fraction_for_adaptive_k(_response(stats)) is None

    def test_no_stats_skips_record(self) -> None:
        # ``GenerationResponse.stats`` is ``None`` for intermediate
        # streaming chunks; only the final response carries stats.
        # Skip silently.
        assert _acceptance_fraction_for_adaptive_k(_response(None)) is None

    def test_zero_acceptance_still_records(self) -> None:
        # An honest 0% acceptance run (drafter ran, target rejected
        # everything) is a valid signal that the drafter is hurting
        # us. Recording it lets adaptive K shrink K toward 1.
        stats = _stats(
            draft_mode="model",
            generation_tokens=20,
            accepted_draft_tokens=0,
            drafter_model_id="mlx-community/test-drafter",
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 0.0)

    def test_full_acceptance_records_one(self) -> None:
        # All generated tokens came from the drafter. Possible in
        # n-gram mode on highly repetitive prompts.
        stats = _stats(
            draft_mode="ngram",
            generation_tokens=12,
            accepted_draft_tokens=12,
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 1.0)

    def test_pipelined_mode_records_accepted_fraction(self) -> None:
        # Codex P2 (PR #20 round-(N+5),
        # batch_generator.py:111-112): asymmetric/pipelined drafting
        # emits ``draft_mode="pipelined"`` with the same
        # ``accepted_draft_tokens`` telemetry as ``model``, but the
        # original gate excluded it from the rolling window. With
        # ``EXO_ADAPTIVE_DRAFT_TOKENS=1`` enabled and asymmetric
        # placement active, ``adaptive_num_draft_tokens`` therefore
        # never adapted -- pinned to the fallback K forever. Verify the
        # gate now feeds pipelined samples into the rolling window.
        stats = _stats(
            draft_mode="pipelined",
            generation_tokens=10,
            accepted_draft_tokens=7,
            drafter_model_id="mlx-community/test-drafter",
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 0.7)

    def test_pipelined_mode_records_zero_acceptance(self) -> None:
        # An honest 0% acceptance run on the pipelined transport (e.g.
        # cold drafter on a new domain) is a valid signal that adaptive
        # K should shrink. Pre-fix this sample never reached the rolling
        # window, so the controller stayed pinned to the fallback even
        # when the drafter was actively hurting throughput.
        stats = _stats(
            draft_mode="pipelined",
            generation_tokens=15,
            accepted_draft_tokens=0,
            drafter_model_id="mlx-community/test-drafter",
        )
        result = _acceptance_fraction_for_adaptive_k(_response(stats))
        assert result is not None
        assert math.isclose(result, 0.0)
