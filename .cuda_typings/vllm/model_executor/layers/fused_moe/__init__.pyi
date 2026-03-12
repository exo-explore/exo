from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
    activation_without_mul as activation_without_mul,
    apply_moe_activation as apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts as BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassBatchedExpertsFp8 as CutlassBatchedExpertsFp8,
    CutlassExpertsFp8 as CutlassExpertsFp8,
    CutlassExpertsW4A8Fp8 as CutlassExpertsW4A8Fp8,
    cutlass_moe_w4a8_fp8 as cutlass_moe_w4a8_fp8,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts as DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts as BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as TritonExperts,
    TritonWNA16Experts as TritonWNA16Experts,
    fused_experts as fused_experts,
    get_config_file_name as get_config_file_name,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoeWeightScaleSupported as FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat as FusedMoEActivationFormat,
    FusedMoEExpertsModular as FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular as FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    AiterExperts as AiterExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter as FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk as fused_topk,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    GateLinear as GateLinear,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk as GroupedTopk,
)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
    SharedFusedMoE as SharedFusedMoE,
)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts as TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.xpu_fused_moe import (
    XPUExperts as XPUExperts,
    XPUExpertsFp8 as XPUExpertsFp8,
)
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE as ZeroExpertFusedMoE,
)

__all__ = [
    "FusedMoE",
    "FusedMoERouter",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "MoEActivation",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEExpertsModular",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalizeModular",
    "GateLinear",
    "RoutingMethodType",
    "SharedFusedMoE",
    "ZeroExpertFusedMoE",
    "activation_without_mul",
    "apply_moe_activation",
    "override_config",
    "get_config",
    "AiterExperts",
    "fused_topk",
    "fused_experts",
    "get_config_file_name",
    "GroupedTopk",
    "cutlass_moe_w4a8_fp8",
    "CutlassExpertsFp8",
    "CutlassBatchedExpertsFp8",
    "CutlassExpertsW4A8Fp8",
    "TritonExperts",
    "TritonWNA16Experts",
    "BatchedTritonExperts",
    "DeepGemmExperts",
    "BatchedDeepGemmExperts",
    "TritonOrDeepGemmExperts",
    "XPUExperts",
    "XPUExpertsFp8",
]

@contextmanager
def override_config(config) -> Generator[None]: ...
def get_config() -> dict[str, Any] | None: ...
