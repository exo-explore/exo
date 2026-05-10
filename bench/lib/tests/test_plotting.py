"""Smoke tests for ``bench.lib.plotting``.

Renders a synthetic benchmark JSON to a tmp PNG and verifies the file is
non-empty. We deliberately don't assert on pixel values — matplotlib
output isn't byte-stable across versions — but a non-empty PNG with a
valid header is a strong signal the renderer didn't throw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from bench.lib.plotting import PlotInputs, render_context_scaling


def _write_synthetic_run(path: Path, *, run_id: str, model: str = "test/model") -> None:
    """Write a minimal context-scaling-shaped JSON for plotting tests."""
    payload = {
        "metadata": {
            "run_id": run_id,
            "benchmark": "context_scaling",
            "started_at": "2026-05-10T00:00:00Z",
            "exo_sha": "deadbeef",
            "hostname": "test-host",
            "platform": "Linux 6.0 (x86_64)",
            "tags": {"operator": "tester"},
        },
        "params": {
            "pp_step": 256,
            "num_steps": 4,
            "tg": 32,
            "warmup": 1,
            "full_model_id": model,
        },
        "cluster": {},
        "runs": [
            {
                "step_index": 0,
                "phase": "cached_sweep",
                "pp_tokens": 256,
                "delta_tokens": 256,
                "prompt_tps": 1800.0,
                "generation_tps": 410.0,
                "prefix_cache_hit": "exact",
                "prompt_tokens": 256,
                "generation_tokens": 32,
                "elapsed_s": 0.14,
                "peak_memory_bytes": 1_000_000_000,
                "output_text_preview": "",
            },
            {
                "step_index": 1,
                "phase": "cached_sweep",
                "pp_tokens": 512,
                "delta_tokens": 256,
                "prompt_tps": 2000.0,
                "generation_tps": 395.0,
                "prefix_cache_hit": "partial",
                "prompt_tokens": 512,
                "generation_tokens": 32,
                "elapsed_s": 0.13,
                "peak_memory_bytes": 1_100_000_000,
                "output_text_preview": "",
            },
            {
                "step_index": 2,
                "phase": "cached_sweep",
                "pp_tokens": 768,
                "delta_tokens": 256,
                "prompt_tps": 2200.0,
                "generation_tps": 378.0,
                "prefix_cache_hit": "partial",
                "prompt_tokens": 768,
                "generation_tokens": 32,
                "elapsed_s": 0.12,
                "peak_memory_bytes": 1_200_000_000,
                "output_text_preview": "",
            },
        ],
        "cold_controls": [
            {
                "phase": "cold_control",
                "pp_tokens": 512,
                "delta_tokens": 512,
                "prompt_tps": 3200.0,
                "generation_tps": 400.0,
                "prefix_cache_hit": "none",
                "prompt_tokens": 512,
                "generation_tokens": 32,
                "elapsed_s": 0.16,
                "peak_memory_bytes": 1_500_000_000,
                "output_text_preview": "",
            },
        ],
        "derived": {
            "t_cum_seconds": [0.14, 0.27, 0.39],
            "control_gaps": [],
        },
    }
    _ = path.write_text(json.dumps(payload))


def _png_is_valid(path: Path) -> bool:
    """A PNG file starts with the 8-byte magic ``\\x89PNG\\r\\n\\x1a\\n``."""
    if not path.is_file():
        return False
    if path.stat().st_size < 100:
        return False
    head = path.read_bytes()[:8]
    return head == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------


class TestRenderContextScaling:
    def test_single_run(self, tmp_path: Path) -> None:
        json_path = tmp_path / "run.json"
        _write_synthetic_run(json_path, run_id="r1")
        out = tmp_path / "out.png"

        returned = render_context_scaling(PlotInputs(results=[json_path], output=out))
        assert returned == out
        assert _png_is_valid(out)

    def test_creates_output_parent_dir(self, tmp_path: Path) -> None:
        json_path = tmp_path / "run.json"
        _write_synthetic_run(json_path, run_id="r1")
        out = tmp_path / "nested" / "deep" / "out.png"

        _ = render_context_scaling(PlotInputs(results=[json_path], output=out))
        assert _png_is_valid(out)

    def test_comparison_two_runs(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_synthetic_run(a, run_id="run-a", model="test/model-a")
        _write_synthetic_run(b, run_id="run-b", model="test/model-b")
        out = tmp_path / "compare.png"

        _ = render_context_scaling(PlotInputs(results=[a, b], output=out))
        assert _png_is_valid(out)

    def test_label_tag_uses_metadata_tag(self, tmp_path: Path) -> None:
        # Smoke test: just confirm passing label_tag doesn't throw and the
        # PNG renders. Label content is too matplotlib-internal to inspect.
        json_path = tmp_path / "run.json"
        _write_synthetic_run(json_path, run_id="r1")
        out = tmp_path / "out.png"

        _ = render_context_scaling(
            PlotInputs(results=[json_path], output=out, label_tag="operator")
        )
        assert _png_is_valid(out)

    def test_explicit_title(self, tmp_path: Path) -> None:
        json_path = tmp_path / "run.json"
        _write_synthetic_run(json_path, run_id="r1")
        out = tmp_path / "out.png"

        _ = render_context_scaling(
            PlotInputs(results=[json_path], output=out, title="Custom Title")
        )
        assert _png_is_valid(out)

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _ = render_context_scaling(
                PlotInputs(results=[], output=tmp_path / "out.png")
            )

    def test_wrong_benchmark_raises(self, tmp_path: Path) -> None:
        # Same shape but with the wrong metadata.benchmark
        json_path = tmp_path / "run.json"
        _write_synthetic_run(json_path, run_id="r1")
        raw_loaded: Any = json.loads(json_path.read_text())  # type: ignore[reportAny]
        assert isinstance(raw_loaded, dict)
        data = cast("dict[str, dict[str, str]]", raw_loaded)
        data["metadata"]["benchmark"] = "something_else"
        _ = json_path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="context_scaling"):
            _ = render_context_scaling(
                PlotInputs(results=[json_path], output=tmp_path / "out.png")
            )
