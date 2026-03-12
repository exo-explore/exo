import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

def is_supported_config_trtllm_bf16(
    moe_config: FusedMoEConfig, activation_format: mk.FusedMoEActivationFormat
) -> tuple[bool, str | None]: ...
def flashinfer_fused_moe_bf16(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor: ...
def flashinfer_fused_moe_bf16_fake(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int = ...,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor: ...
