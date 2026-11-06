import time
from collections.abc import Sequence

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel


class TokenPacket(FrozenModel):
    """Encapsulates a single token's hidden state and routing metadata for MoE dispatch."""

    token_id: int
    sequence_id: str
    hidden_states: tuple[float, ...]
    target_expert_id: int
    source_node_id: str
    routing_weight: float = 1.0
    layer_index: int = 0


class DispatchBatch(FrozenModel):
    """Batched grouping of token packets destined for a common target expert and node."""

    batch_id: str
    target_node_id: str
    target_expert_id: int
    packets: Sequence[TokenPacket] = Field(default_factory=tuple)
    creation_timestamp: float = Field(default_factory=time.time)

    @property
    def token_count(self) -> int:
        return len(self.packets)

    @property
    def payload_bytes_estimate(self) -> int:
        # Approximate 4 bytes per float32 plus packet metadata
        float_bytes = sum(len(p.hidden_states) * 4 for p in self.packets)
        metadata_bytes = len(self.packets) * 64
        return float_bytes + metadata_bytes


class AllToAllTelemetry(FrozenModel):
    """Telemetry metrics tracking asynchronous All-to-All communication overlapping."""

    dispatch_id: str
    tokens_dispatched: int
    batches_count: int
    total_transmission_time_ms: float
    computation_overlap_time_ms: float
    effective_bandwidth_mbps: float = 0.0

    @property
    def overlap_ratio(self) -> float:
        """Percentage of transmission latency hidden behind local computation (0.0 to 1.0)."""
        if self.total_transmission_time_ms <= 0:
            return 1.0
        return min(1.0, self.computation_overlap_time_ms / self.total_transmission_time_ms)
