import numpy as np
from numpy.typing import NDArray

from exo.shared.types.moe_fallback import (
    ElasticMoEConfig,
    ExpertExecutionResult,
    MoEFallbackDecision,
    MoEFallbackTelemetry,
)

EPSILON: float = 1e-9


class ElasticMoEFallbackExecutor:
    """Executes fault-tolerant MoE expert aggregation with dynamic weight redistribution and residual bypass."""

    def __init__(self, config: ElasticMoEConfig | None = None) -> None:
        self.config = config or ElasticMoEConfig()

    def aggregate_with_fallback(
        self,
        results: list[ExpertExecutionResult],
        expert_tensors: dict[int, NDArray[np.floating]],
        residual_input: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.float32], MoEFallbackDecision, MoEFallbackTelemetry]:
        """Aggregates MoE expert outputs, redistributing weights for dropped/timed-out nodes.

        Args:
            results: Status and original gating weights for each dispatched expert.
            expert_tensors: Mapping of expert_id to their computed output tensor.
            residual_input: Input activations prior to MoE, used as bypass fallback if all experts fail.

        Returns:
            Tuple of (aggregated_moe_output, MoEFallbackDecision, MoEFallbackTelemetry).
        """
        successful_experts: list[int] = []
        skipped_experts: list[int] = []
        timeout_count: int = 0
        disconnected_count: int = 0

        for r in results:
            if r.status == "SUCCESS" and r.expert_id in expert_tensors:
                successful_experts.append(r.expert_id)
            else:
                skipped_experts.append(r.expert_id)
                if r.status == "TIMEOUT":
                    timeout_count += 1
                elif r.status == "DISCONNECTED":
                    disconnected_count += 1

        total_experts_queried = len(results)
        total_original_weight = sum(r.original_weight for r in results)
        successful_original_weight = sum(
            r.original_weight for r in results if r.expert_id in successful_experts
        )
        lost_weight = total_original_weight - successful_original_weight
        lost_fraction = (
            lost_weight / total_original_weight if total_original_weight > 0 else 0.0
        )

        # Case 1: At least one expert succeeded and satisfies threshold
        if len(successful_experts) >= self.config.min_successful_experts:
            redistributed_weights: dict[int, float] = {}

            if successful_original_weight > EPSILON:
                # Re-normalize successful weights so their sum is 1.0
                target_sum: float = 1.0
                scale_factor = target_sum / successful_original_weight
                for r in results:
                    if r.expert_id in successful_experts:
                        redistributed_weights[r.expert_id] = r.original_weight * scale_factor
            else:
                equal_w = 1.0 / len(successful_experts)
                for eid in successful_experts:
                    redistributed_weights[eid] = equal_w

            # Compute weighted sum
            first_eid = successful_experts[0]
            first_tensor = np.asarray(expert_tensors[first_eid], dtype=np.float32)
            aggregated = np.zeros_like(first_tensor)

            for eid in successful_experts:
                w = redistributed_weights[eid]
                t = np.asarray(expert_tensors[eid], dtype=np.float32)
                aggregated += w * t

            decision = MoEFallbackDecision(
                successful_experts=successful_experts,
                skipped_experts=skipped_experts,
                redistributed_weights=redistributed_weights,
                residual_bypass_triggered=False,
            )
            telemetry = MoEFallbackTelemetry(
                total_experts_queried=total_experts_queried,
                successful_count=len(successful_experts),
                timeout_count=timeout_count,
                disconnected_count=disconnected_count,
                lost_weight_fraction=lost_fraction,
                residual_bypass_triggered=False,
            )
            return aggregated, decision, telemetry

        # Case 2: Extreme failure - all experts timed out or disconnected
        residual_bypass_triggered = True
        redistributed_weights = {}

        if residual_input is not None:
            # Bypass MoE: return zeros so that Activation + MoE(0) = Activation
            aggregated = np.zeros_like(residual_input, dtype=np.float32)
        else:
            aggregated = np.zeros((1, 1), dtype=np.float32)

        decision = MoEFallbackDecision(
            successful_experts=[],
            skipped_experts=skipped_experts,
            redistributed_weights=redistributed_weights,
            residual_bypass_triggered=residual_bypass_triggered,
        )
        telemetry = MoEFallbackTelemetry(
            total_experts_queried=total_experts_queried,
            successful_count=0,
            timeout_count=timeout_count,
            disconnected_count=disconnected_count,
            lost_weight_fraction=1.0,
            residual_bypass_triggered=residual_bypass_triggered,
        )
        return aggregated, decision, telemetry
