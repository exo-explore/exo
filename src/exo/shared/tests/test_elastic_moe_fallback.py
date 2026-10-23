import numpy as np
import pytest

from exo.shared.types.moe_fallback import (
    ElasticMoEConfig,
    ExpertExecutionResult,
)
from exo.worker.engines.moe.elastic_fallback_executor import ElasticMoEFallbackExecutor


def test_all_experts_successful() -> None:
    executor = ElasticMoEFallbackExecutor()
    t0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    results = [
        ExpertExecutionResult(expert_id=0, status="SUCCESS", original_weight=0.6),
        ExpertExecutionResult(expert_id=1, status="SUCCESS", original_weight=0.4),
    ]
    tensors = {0: t0, 1: t1}

    output, decision, telemetry = executor.aggregate_with_fallback(results, tensors)

    expected = 0.6 * t0 + 0.4 * t1
    np.testing.assert_allclose(output, expected, rtol=1e-5)
    assert decision.successful_experts == [0, 1]
    assert decision.skipped_experts == []
    assert decision.residual_bypass_triggered is False
    assert telemetry.lost_weight_fraction == 0.0
    assert telemetry.successful_count == 2
    assert decision.redistributed_weights[0] == pytest.approx(0.6)
    assert decision.redistributed_weights[1] == pytest.approx(0.4)


def test_single_expert_timeout_redistribution() -> None:
    executor = ElasticMoEFallbackExecutor()
    t0 = np.array([[2.0, 4.0]], dtype=np.float32)

    results = [
        ExpertExecutionResult(expert_id=0, status="SUCCESS", original_weight=0.7),
        ExpertExecutionResult(expert_id=1, status="TIMEOUT", original_weight=0.3),
    ]
    tensors = {0: t0}

    output, decision, telemetry = executor.aggregate_with_fallback(results, tensors)

    # Expert 0 absorbs the remaining fraction and scales to 1.0
    np.testing.assert_allclose(output, t0, rtol=1e-5)
    assert decision.successful_experts == [0]
    assert decision.skipped_experts == [1]
    assert decision.redistributed_weights[0] == pytest.approx(1.0)
    assert decision.residual_bypass_triggered is False
    assert telemetry.timeout_count == 1
    assert telemetry.lost_weight_fraction == pytest.approx(0.3)


def test_multiple_experts_partial_disconnection() -> None:
    executor = ElasticMoEFallbackExecutor()
    t0 = np.ones((2, 2), dtype=np.float32) * 2.0
    t2 = np.ones((2, 2), dtype=np.float32) * 5.0

    results = [
        ExpertExecutionResult(expert_id=0, status="SUCCESS", original_weight=0.5),
        ExpertExecutionResult(expert_id=1, status="DISCONNECTED", original_weight=0.3),
        ExpertExecutionResult(expert_id=2, status="SUCCESS", original_weight=0.2),
    ]
    tensors = {0: t0, 2: t2}

    output, decision, telemetry = executor.aggregate_with_fallback(results, tensors)

    # Weights 0.5 and 0.2 re-normalized over 0.7
    w0 = 0.5 / 0.7
    w2 = 0.2 / 0.7
    expected = w0 * t0 + w2 * t2

    np.testing.assert_allclose(output, expected, rtol=1e-5)
    assert decision.redistributed_weights[0] == pytest.approx(w0)
    assert decision.redistributed_weights[2] == pytest.approx(w2)
    assert decision.skipped_experts == [1]
    assert telemetry.disconnected_count == 1
    assert telemetry.lost_weight_fraction == pytest.approx(0.3)


def test_all_experts_failed_residual_bypass() -> None:
    config = ElasticMoEConfig(enable_residual_bypass=True, min_successful_experts=1)
    executor = ElasticMoEFallbackExecutor(config)
    residual_input = np.ones((3, 8), dtype=np.float32) * 3.14

    results = [
        ExpertExecutionResult(expert_id=0, status="TIMEOUT", original_weight=0.5),
        ExpertExecutionResult(expert_id=1, status="DISCONNECTED", original_weight=0.5),
    ]
    tensors: dict[int, np.ndarray] = {}

    output, decision, telemetry = executor.aggregate_with_fallback(
        results, tensors, residual_input=residual_input
    )

    # Output should be zeros with matching shape, ensuring (Residual + Output) == Residual
    assert output.shape == residual_input.shape
    np.testing.assert_array_equal(output, np.zeros_like(residual_input))
    assert decision.residual_bypass_triggered is True
    assert decision.successful_experts == []
    assert decision.skipped_experts == [0, 1]
    assert telemetry.lost_weight_fraction == 1.0
    assert telemetry.timeout_count == 1
    assert telemetry.disconnected_count == 1
