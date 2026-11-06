from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

ExpertStatus = Literal["SUCCESS", "TIMEOUT", "DISCONNECTED"]


class ElasticMoEConfig(FrozenModel):
    """Configuration for edge MoE fault-tolerant execution and early-exit fallback."""

    expert_timeout_ms: float = 100.0
    enable_residual_bypass: bool = True
    min_successful_experts: int = 1


class ExpertExecutionResult(FrozenModel):
    """Result of an asynchronous expert inference dispatch."""

    expert_id: int
    status: ExpertStatus
    output_bytes: bytes = b""
    original_weight: float = 0.0
    latency_ms: float = 0.0


class MoEFallbackDecision(FrozenModel):
    """Routing and aggregation decision after handling expert timeouts or disconnections."""

    successful_experts: list[int]
    skipped_experts: list[int]
    redistributed_weights: dict[int, float]
    residual_bypass_triggered: bool


class MoEFallbackTelemetry(FrozenModel):
    """Telemetry report capturing MoE fault-tolerance metrics and degraded aggregation status."""

    total_experts_queried: int
    successful_count: int
    timeout_count: int
    disconnected_count: int
    lost_weight_fraction: float
    residual_bypass_triggered: bool
