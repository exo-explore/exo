import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.platforms import current_platform as current_platform

class XPUExperts(mk.FusedMoEExpertsModular):
    is_fp8: bool
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None: ...
    @property
    def expects_unquantized_inputs(self) -> bool: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...

class XPUExpertsFp8(XPUExperts):
    is_fp8: bool
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None: ...
