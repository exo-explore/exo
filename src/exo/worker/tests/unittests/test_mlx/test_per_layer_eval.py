import pytest

from .conftest import (
    DEFAULT_GPT_OSS_CONFIG,
)
from .test_distributed_fix import run_tensor_test


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


pytestmark = [
    pytest.mark.skipif(
        not _check_model_exists(),
        reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
    ),
]


class TestPerLayerEvaluation:
    """Test per-layer evaluation strategy."""

    def test_per_layer_eval_completes(self) -> None:
        """Test that per-layer eval completes without deadlock."""
        result = run_tensor_test(
            prompt_tokens=100,
            prefill_step_size=64,
            port_offset=700,
            process_timeout=120,
        )
        assert not result.timed_out, "Per-layer eval should complete without timeout"
        assert result.all_success, f"Failures: {result.results}"

    def test_per_layer_eval_with_large_prompt(self) -> None:
        """Test per-layer eval with larger prompt."""
        result = run_tensor_test(
            prompt_tokens=500,
            prefill_step_size=256,
            port_offset=800,
            process_timeout=180,
        )
        assert not result.timed_out, "Per-layer eval should handle large prompts"
        assert result.all_success, f"Failures: {result.results}"


class TestTensorParallelWithFastSynch:
    """Test tensor parallel with FAST_SYNCH=1 using per-layer eval."""

    def test_tensor_parallel_fast_synch_no_deadlock(self) -> None:
        """Test that per-layer eval prevents FAST_SYNCH deadlock.

        This test verifies the per-layer evaluation strategy works
        with the default FAST_SYNCH behavior (auto-enabled for JACCL).
        """
        result = run_tensor_test(
            prompt_tokens=100,
            prefill_step_size=64,
            port_offset=900,
            process_timeout=120,
        )
        assert not result.timed_out, "Per-layer eval should prevent FAST_SYNCH deadlock"
        assert result.all_success, f"Failures: {result.results}"
