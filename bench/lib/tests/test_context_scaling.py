"""Unit tests for the pure helpers in ``bench.lib.context_scaling``.

The orchestration entry points (``run``, ``run_cached_sweep``,
``run_cold_controls``, ``make_cold_control_factory``) need a real
``BenchSession`` and exo cluster, so they're exercised end-to-end via
``python -m bench.cli context-scaling``. This module covers the
underscore-prefixed pure helpers via direct private-symbol access (the
private prefix discourages library users; tests for those helpers are
the explicit exception).
"""

from __future__ import annotations

import math
from typing import cast

from bench.lib.context_scaling import (
    StepResult,
    _compute_t_cum,  # type: ignore[reportPrivateUsage]
    _interp,  # type: ignore[reportPrivateUsage]
    derive_summary,
)


def _step(
    *,
    pp: int,
    delta: int,
    prompt_tps: float,
    generation_tps: float = 100.0,
    hit: str = "partial",
) -> StepResult:
    return StepResult(
        pp_tokens=pp,
        delta_tokens=delta,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prefix_cache_hit=hit,
        prompt_tokens=pp,
        generation_tokens=32,
        elapsed_s=delta / prompt_tps if prompt_tps else 0.0,
    )


def _close(actual: float, expected: float, abs_tol: float = 1e-3) -> bool:
    return math.isclose(actual, expected, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# _compute_t_cum
# ---------------------------------------------------------------------------


class TestComputeTCum:
    def test_empty_returns_empty(self) -> None:
        assert _compute_t_cum([]) == []

    def test_single_step(self) -> None:
        # 256 tokens at 1024 tps -> 0.25s
        out = _compute_t_cum([_step(pp=256, delta=256, prompt_tps=1024.0)])
        assert len(out) == 1
        assert _close(out[0], 0.25)

    def test_cumulative_sum_across_three_steps(self) -> None:
        steps = [
            _step(pp=256, delta=256, prompt_tps=1000.0),  # 0.256s
            _step(pp=512, delta=256, prompt_tps=2000.0),  # +0.128s = 0.384s
            _step(pp=768, delta=256, prompt_tps=512.0),  # +0.500s = 0.884s
        ]
        out = _compute_t_cum(steps)
        assert _close(out[0], 0.256)
        assert _close(out[1], 0.384)
        assert _close(out[2], 0.884)
        # Monotonically non-decreasing
        assert out == sorted(out)

    def test_zero_tps_step_skipped(self) -> None:
        # A row with prompt_tps == 0 contributes nothing to the cumulative sum
        steps = [
            _step(pp=256, delta=256, prompt_tps=1024.0),  # +0.25s
            _step(pp=512, delta=256, prompt_tps=0.0),  # +0
            _step(pp=768, delta=256, prompt_tps=512.0),  # +0.5s
        ]
        out = _compute_t_cum(steps)
        assert _close(out[0], 0.25)
        assert _close(out[1], 0.25)  # unchanged
        assert _close(out[2], 0.75)

    def test_zero_delta_step_skipped(self) -> None:
        # Defensive: a Δ=0 row would otherwise add zero anyway, but we
        # explicitly guard against negative delta + 0/0.
        steps = [
            _step(pp=256, delta=256, prompt_tps=1000.0),
            _step(pp=256, delta=0, prompt_tps=1000.0),  # explicit Δ=0
        ]
        out = _compute_t_cum(steps)
        assert _close(out[0], 0.256)
        assert _close(out[1], 0.256)


# ---------------------------------------------------------------------------
# _interp
# ---------------------------------------------------------------------------


class TestInterp:
    def test_empty_points_returns_zero(self) -> None:
        assert _interp([], 100) == 0.0

    def test_single_point_returns_y(self) -> None:
        assert _interp([(100, 1.5)], 50) == 1.5
        assert _interp([(100, 1.5)], 100) == 1.5
        assert _interp([(100, 1.5)], 200) == 1.5

    def test_clamps_below_first(self) -> None:
        points = [(100, 0.1), (200, 0.3), (300, 0.6)]
        assert _interp(points, 0) == 0.1
        assert _interp(points, 50) == 0.1
        assert _interp(points, 100) == 0.1

    def test_clamps_above_last(self) -> None:
        points = [(100, 0.1), (200, 0.3), (300, 0.6)]
        assert _interp(points, 300) == 0.6
        assert _interp(points, 500) == 0.6
        assert _interp(points, 1_000_000) == 0.6

    def test_mid_bracket_linear_interpolation(self) -> None:
        points = [(100, 0.0), (200, 1.0)]
        assert _close(_interp(points, 150), 0.5)
        assert _close(_interp(points, 175), 0.75)

    def test_multi_segment_linear_interpolation(self) -> None:
        # Two adjacent segments, x=250 falls in the second one
        points = [(100, 0.1), (200, 0.3), (300, 0.6)]
        # 200..300: 0.3 + (0.6-0.3) * (250-200)/(300-200) = 0.3 + 0.15 = 0.45
        assert _close(_interp(points, 250), 0.45)


# ---------------------------------------------------------------------------
# derive_summary
# ---------------------------------------------------------------------------


def _gap_at(summary: dict[str, object], index: int) -> dict[str, float]:
    """Cast ``summary['control_gaps'][index]`` into the typed shape we expect."""
    raw = summary["control_gaps"]
    assert isinstance(raw, list)
    entry = cast("dict[str, float]", raw[index])
    return entry


class TestDeriveSummary:
    def test_no_controls_only_t_cum(self) -> None:
        steps = [
            _step(pp=256, delta=256, prompt_tps=1024.0),
            _step(pp=512, delta=256, prompt_tps=1024.0),
        ]
        summary = derive_summary(steps, [])
        t_cum = cast("list[float]", summary["t_cum_seconds"])
        assert _close(t_cum[0], 0.25)
        assert _close(t_cum[1], 0.5)
        assert summary["control_gaps"] == []

    def test_control_gap_at_known_pp(self) -> None:
        # Sweep: 0.25s @ pp=256, 0.5s @ pp=512
        steps = [
            _step(pp=256, delta=256, prompt_tps=1024.0),
            _step(pp=512, delta=256, prompt_tps=1024.0),
        ]
        # Cold control at pp=512, 2x faster than the per-step rate -> 0.25s
        controls = [_step(pp=512, delta=512, prompt_tps=2048.0, hit="none")]
        summary = derive_summary(steps, controls)
        gap = _gap_at(summary, 0)
        assert gap["pp_tokens"] == 512
        assert _close(gap["cold_t_seconds"], 0.25, abs_tol=0.01)
        assert _close(gap["t_cum_seconds_at_pp"], 0.5, abs_tol=0.01)
        assert _close(gap["gap_seconds"], 0.25, abs_tol=0.01)
        # gap_fraction = 0.25 / 0.25 = 1.0
        assert _close(gap["gap_fraction"], 1.0, abs_tol=0.01)

    def test_control_gap_zero_cold_tps_yields_zero_fraction(self) -> None:
        steps = [_step(pp=256, delta=256, prompt_tps=1000.0)]
        controls = [_step(pp=256, delta=256, prompt_tps=0.0, hit="none")]
        gap = _gap_at(derive_summary(steps, controls), 0)
        assert gap["cold_t_seconds"] == 0.0
        assert gap["gap_fraction"] == 0.0
