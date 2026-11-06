from typing import Any, Literal

from pydantic import ConfigDict, Field

from exo.utils.pydantic_ext import FrozenModel

NpuBackendType = Literal["directml", "qnn", "openvino", "cpu_fallback"]
OperatorKind = Literal["rmsnorm", "linear", "silu", "gelu", "matmul", "add", "rope", "softmax"]

# Operators inherently suited for fixed-topology tensor accelerators (NPUs)
NPU_FUSIBLE_OPERATORS: frozenset[OperatorKind] = frozenset(
    {"rmsnorm", "linear", "silu", "gelu", "matmul", "add"}
)


class ComputationNode(FrozenModel):
    """Represents a discrete mathematical operation node in a computation graph."""

    model_config = ConfigDict(frozen=True, strict=True)

    node_id: str
    op_type: OperatorKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    is_npu_supported: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class FusedSubgraphDescriptor(FrozenModel):
    """Represents a fused chain of operators compiled for execution on an NPU accelerator or host."""

    model_config = ConfigDict(frozen=True, strict=True)

    subgraph_id: str
    target_backend: NpuBackendType
    nodes: tuple[ComputationNode, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    is_fused_npu: bool = True

    @property
    def num_ops(self) -> int:
        return len(self.nodes)


class NpuSessionTelemetry(FrozenModel):
    """Telemetry capturing NPU graph fusion execution metrics and offload ratios."""

    model_config = ConfigDict(frozen=True, strict=True)

    backend_used: NpuBackendType
    total_subgraphs_executed: int = 0
    fused_npu_subgraphs_count: int = 0
    cpu_fallback_subgraphs_count: int = 0
    total_npu_execution_time_ms: float = 0.0
    total_cpu_fallback_time_ms: float = 0.0

    @property
    def npu_offload_ratio(self) -> float:
        total = self.fused_npu_subgraphs_count + self.cpu_fallback_subgraphs_count
        if total == 0:
            return 0.0
        return self.fused_npu_subgraphs_count / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_used": self.backend_used,
            "total_subgraphs_executed": self.total_subgraphs_executed,
            "fused_npu_subgraphs_count": self.fused_npu_subgraphs_count,
            "cpu_fallback_subgraphs_count": self.cpu_fallback_subgraphs_count,
            "npu_offload_ratio_pct": round(self.npu_offload_ratio * 100.0, 2),
            "total_npu_execution_time_ms": round(self.total_npu_execution_time_ms, 3),
            "total_cpu_fallback_time_ms": round(self.total_cpu_fallback_time_ms, 3),
        }
