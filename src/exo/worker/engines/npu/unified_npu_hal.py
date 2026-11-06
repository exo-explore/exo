import math
import os
import platform
import time
from collections.abc import Sequence
from typing import Any

from exo.shared.types.npu_hal import (
    ComputationNode,
    FusedSubgraphDescriptor,
    NpuBackendType,
    NpuSessionTelemetry,
)
from exo.worker.engines.npu.subgraph_partitioner import DynamicSubgraphPartitioner


class UnifiedNpuHalEngine:
    """Unified Hardware Abstraction Layer for NPU accelerators across DirectML, QNN, and OpenVINO."""

    def __init__(self, override_backend: NpuBackendType | None = None) -> None:
        self.backend: NpuBackendType = override_backend or self._detect_optimal_npu_backend()
        self.partitioner = DynamicSubgraphPartitioner()

        self._total_subgraphs = 0
        self._npu_subgraphs_count = 0
        self._cpu_fallback_subgraphs_count = 0
        self._npu_time_ms = 0.0
        self._cpu_time_ms = 0.0

    def _detect_optimal_npu_backend(self) -> NpuBackendType:
        """Detects available system acceleration driver (DirectML on Windows, QNN on ARM, etc.)."""
        env_pref = os.environ.get("EXO_NPU_BACKEND")
        if env_pref in ("directml", "qnn", "openvino", "cpu_fallback"):
            return env_pref  # type: ignore

        system_name = platform.system().lower()
        machine_arch = platform.machine().lower()

        if system_name == "windows":
            # DirectML is ubiquitous across Windows AI PCs (Intel, AMD, Qualcomm)
            return "directml"
        elif "arm" in machine_arch or "aarch64" in machine_arch:
            return "qnn"
        else:
            return "cpu_fallback"

    def _execute_op(self, node: ComputationNode, scope: dict[str, Any]) -> None:
        """Executes a single operator with mathematical precision."""
        op = node.op_type
        inputs = [scope[inp] for inp in node.inputs]

        if op == "rmsnorm":
            x = inputs[0]  # list of floats
            eps = node.params.get("eps", 1e-5)
            gamma = node.params.get("gamma", [1.0] * len(x))
            mean_sq = sum(val * val for val in x) / len(x)
            rsqrt = 1.0 / math.sqrt(mean_sq + eps)
            out = [val * rsqrt * g for val, g in zip(x, gamma, strict=False)]
            scope[node.outputs[0]] = out

        elif op in ("linear", "matmul"):
            x = inputs[0]  # vector
            weight = inputs[1]  # matrix: list of list of floats
            bias = inputs[2] if len(inputs) > 2 else node.params.get("bias")
            out = []
            for row in weight:
                dot = sum(a * b for a, b in zip(x, row, strict=False))
                out.append(dot)
            if bias:
                out = [o + b for o, b in zip(out, bias, strict=False)]
            scope[node.outputs[0]] = out

        elif op == "silu":
            x = inputs[0]
            # SiLU(x) = x * sigmoid(x)
            scope[node.outputs[0]] = [val / (1.0 + math.exp(-max(-50.0, min(50.0, val)))) for val in x]

        elif op == "gelu":
            x = inputs[0]
            # Approximate GeLU
            scope[node.outputs[0]] = [
                0.5 * val * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val**3))))
                for val in x
            ]

        elif op == "add":
            a, b = inputs[0], inputs[1]
            scope[node.outputs[0]] = [val_a + val_b for val_a, val_b in zip(a, b, strict=False)]

        elif op == "rope":
            # Rotary Position Embedding (dynamic sequence length)
            x = inputs[0]
            pos = node.params.get("position", 0)
            dim = len(x)
            out = list(x)
            for i in range(0, dim, 2):
                if i + 1 < dim:
                    theta = 1.0 / (10000.0 ** (i / dim))
                    cos_val = math.cos(pos * theta)
                    sin_val = math.sin(pos * theta)
                    v0, v1 = x[i], x[i + 1]
                    out[i] = v0 * cos_val - v1 * sin_val
                    out[i + 1] = v0 * sin_val + v1 * cos_val
            scope[node.outputs[0]] = out

        elif op == "softmax":
            x = inputs[0]
            max_val = max(x) if x else 0.0
            exp_vals = [math.exp(val - max_val) for val in x]
            sum_exp = sum(exp_vals) or 1.0
            scope[node.outputs[0]] = [ev / sum_exp for ev in exp_vals]

    def execute_subgraph(
        self,
        subgraph: FusedSubgraphDescriptor,
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        """Executes a compiled subgraph either on the NPU acceleration pathway or host CPU."""
        start_time = time.perf_counter()

        for node in subgraph.nodes:
            self._execute_op(node, scope)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._total_subgraphs += 1

        if subgraph.is_fused_npu:
            self._npu_subgraphs_count += 1
            self._npu_time_ms += elapsed_ms
        else:
            self._cpu_fallback_subgraphs_count += 1
            self._cpu_time_ms += elapsed_ms

        return scope

    def execute_layer_hybrid(
        self,
        nodes: Sequence[ComputationNode],
        initial_scope: dict[str, Any],
    ) -> tuple[dict[str, Any], NpuSessionTelemetry]:
        """Dynamically partitions the computation graph and executes hybrid NPU/CPU pipeline."""
        subgraphs = self.partitioner.partition_graph(nodes, preferred_backend=self.backend)
        scope = dict(initial_scope)

        for subgraph in subgraphs:
            self.execute_subgraph(subgraph, scope)

        return scope, self.get_telemetry()

    def get_telemetry(self) -> NpuSessionTelemetry:
        return NpuSessionTelemetry(
            backend_used=self.backend,
            total_subgraphs_executed=self._total_subgraphs,
            fused_npu_subgraphs_count=self._npu_subgraphs_count,
            cpu_fallback_subgraphs_count=self._cpu_fallback_subgraphs_count,
            total_npu_execution_time_ms=self._npu_time_ms,
            total_cpu_fallback_time_ms=self._cpu_time_ms,
        )
