"""Unit tests for ``bench.lib.model_meta``.

These exercise the pure derivation helpers (no HF round-trip). The HTTP
fetchers (``fetch_model_meta``, ``_read_config_json``, ``_sum_weight_sizes``)
hit the public hub and aren't covered here.
"""

from __future__ import annotations

import math

import pytest

from bench.lib.model_meta import (
    ModelMeta,
    derive_cold_controls,
    derive_context_ramp,
)


def _meta(
    *,
    weight_bytes: int = 0,
    max_pos: int = 4096,
    layers: int = 32,
) -> ModelMeta:
    return ModelMeta(
        model_id="test/model",
        total_weight_bytes=weight_bytes,
        max_position_embeddings=max_pos,
        num_hidden_layers=layers,
    )


# ---------------------------------------------------------------------------
# ModelMeta properties
# ---------------------------------------------------------------------------


class TestModelMetaConstraints:
    def test_zero_weight_yields_one_gib_floor(self) -> None:
        meta = _meta(weight_bytes=0)
        # int(0 * 1.30) + 1 == 1; int(0 * 1.10) + 1 == 1
        assert meta.memory_constraint_gb == 1.0
        assert meta.disk_constraint_gb == 1.0

    def test_one_gib_weight_rounds_up(self) -> None:
        meta = _meta(weight_bytes=1 * (1024**3))
        # int(1.0 * 1.30) + 1 = 2; int(1.0 * 1.10) + 1 = 2
        assert meta.memory_constraint_gb == 2.0
        assert meta.disk_constraint_gb == 2.0

    def test_sixteen_gib_weight_uses_30pct_memory_10pct_disk(self) -> None:
        meta = _meta(weight_bytes=16 * (1024**3))
        # memory: int(16 * 1.30) + 1 = 21; disk: int(16 * 1.10) + 1 = 18
        assert meta.memory_constraint_gb == 21.0
        assert meta.disk_constraint_gb == 18.0

    def test_total_weight_gb_property(self) -> None:
        meta = _meta(weight_bytes=2_147_483_648)  # 2 GiB exactly
        assert math.isclose(meta.total_weight_gb, 2.0)


# ---------------------------------------------------------------------------
# derive_context_ramp
# ---------------------------------------------------------------------------


class TestDeriveContextRamp:
    def test_full_max_evenly_divides_round_to(self) -> None:
        meta = _meta(max_pos=131072)  # 128k
        pp_step, num_steps = derive_context_ramp(meta, num_steps=32)
        # 131072 // 32 = 4096; rounded down to multiple of 256 = 4096
        assert pp_step == 4096
        assert num_steps == 32
        # Top of ramp == max
        assert pp_step * num_steps == 131072

    def test_qwen30b_a3b_ramp(self) -> None:
        meta = _meta(max_pos=40960)  # Qwen3-30B-A3B
        pp_step, num_steps = derive_context_ramp(meta, num_steps=32)
        # 40960 // 32 = 1280; multiple of 256
        assert pp_step == 1280
        assert pp_step * num_steps == 40960

    def test_fraction_of_max_half(self) -> None:
        meta = _meta(max_pos=131072)
        pp_step, num_steps = derive_context_ramp(meta, num_steps=8, fraction_of_max=0.5)
        # half = 65536; 65536 // 8 = 8192
        assert pp_step == 8192
        assert num_steps == 8

    def test_min_pp_step_floor(self) -> None:
        meta = _meta(max_pos=512)
        # 512 // 32 = 16, but min_pp_step=256 floors it; rounded to 256
        pp_step, num_steps = derive_context_ramp(meta, num_steps=32)
        assert pp_step == 256
        assert num_steps == 32

    def test_round_to_truncates_down(self) -> None:
        meta = _meta(max_pos=10000)
        pp_step, _ = derive_context_ramp(meta, num_steps=32, round_to=256)
        # 10000 // 32 = 312; (312 // 256) * 256 = 256
        assert pp_step == 256

    def test_round_to_zero_step_falls_back_to_round_to(self) -> None:
        # Pathological: huge round_to relative to per-step size
        meta = _meta(max_pos=1024)
        pp_step, _ = derive_context_ramp(meta, num_steps=8, round_to=1024)
        # 1024 // 8 = 128, but min_pp_step=256 → 256; (256 // 1024) * 1024 = 0;
        # `or round_to` rescues to 1024.
        assert pp_step == 1024

    def test_max_pos_zero_raises(self) -> None:
        meta = _meta(max_pos=0)
        with pytest.raises(ValueError, match="max_position_embeddings=0"):
            _ = derive_context_ramp(meta, num_steps=32)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, 2.0])
    def test_fraction_outside_unit_interval_raises(self, fraction: float) -> None:
        meta = _meta(max_pos=4096)
        with pytest.raises(ValueError, match="fraction_of_max"):
            _ = derive_context_ramp(meta, num_steps=4, fraction_of_max=fraction)

    @pytest.mark.parametrize("steps", [0, -1, -100])
    def test_num_steps_must_be_positive(self, steps: int) -> None:
        meta = _meta(max_pos=4096)
        with pytest.raises(ValueError, match="num_steps"):
            _ = derive_context_ramp(meta, num_steps=steps)


# ---------------------------------------------------------------------------
# derive_cold_controls
# ---------------------------------------------------------------------------


class TestDeriveColdControls:
    def test_count_zero_returns_empty_tuple(self) -> None:
        meta = _meta()
        assert derive_cold_controls(meta, pp_step=4096, num_steps=32, count=0) == ()

    def test_count_one_returns_top_only(self) -> None:
        meta = _meta()
        assert derive_cold_controls(meta, pp_step=4096, num_steps=32, count=1) == (
            131072,
        )

    def test_evenly_spaced_four(self) -> None:
        meta = _meta()
        out = derive_cold_controls(meta, pp_step=4096, num_steps=32, count=4)
        # max_pp = 131072; (131072 * (i+1)) // 4 for i in {0,1,2,3}
        # = {32768, 65536, 98304, 131072}
        assert out == (32768, 65536, 98304, 131072)

    def test_filters_below_pp_step(self) -> None:
        meta = _meta()
        out = derive_cold_controls(meta, pp_step=8192, num_steps=2, count=4)
        # max_pp = 16384; spaced points = {4096, 8192, 12288, 16384};
        # 4096 < pp_step=8192 → dropped.
        assert out == (8192, 12288, 16384)

    def test_dedups_at_low_count_high_step(self) -> None:
        meta = _meta()
        # max_pp = 1024; count=2 → spaced = {512, 1024}; 512 < pp_step? No (=).
        out = derive_cold_controls(meta, pp_step=512, num_steps=2, count=2)
        assert out == (512, 1024)

    def test_returned_in_ascending_order(self) -> None:
        meta = _meta()
        out = derive_cold_controls(meta, pp_step=1024, num_steps=8, count=4)
        assert list(out) == sorted(out)
