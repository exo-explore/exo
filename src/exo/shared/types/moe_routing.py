from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel


class NodeComputeProfile(FrozenModel):
    """Dynamic profile capturing real-time edge node compute performance and queue depth."""

    node_id: NodeId
    compute_weight: float = 1.0
    queue_depth: int = 0
    is_throttled: bool = False
    max_capacity: int = 128


class ExpertPlacement(FrozenModel):
    """Mapping of an MoE expert to its host edge node."""

    expert_id: int
    node_id: NodeId


class MoERoutingDecision(FrozenModel):
    """Output decision of the device-aware MoE router."""

    selected_experts: list[list[int]]
    routing_weights: list[list[float]]
    overflow_tokens: list[int]
    expert_token_counts: dict[int, int]
    node_token_counts: dict[str, int]


class MoERoutingTelemetry(FrozenModel):
    """Telemetry report capturing MoE load balancing efficiency across heterogeneous nodes."""

    total_tokens: int
    num_experts: int
    overflow_count: int
    overflow_rate: float
    load_balance_cv: float
