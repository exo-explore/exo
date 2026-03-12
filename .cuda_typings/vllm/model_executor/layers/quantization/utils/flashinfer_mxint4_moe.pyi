import functools
import torch

__all__ = [
    "prepare_static_weights_for_trtllm_mxint4_moe",
    "flashinfer_trtllm_mxint4_moe",
    "is_flashinfer_mxint4_moe_available",
]

@functools.cache
def is_flashinfer_mxint4_moe_available() -> bool: ...
def prepare_static_weights_for_trtllm_mxint4_moe(
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
) -> dict[str, torch.Tensor]: ...
def flashinfer_trtllm_mxint4_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight_packed: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_packed: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    global_num_experts: int,
    top_k: int,
    intermediate_size_per_partition: int,
    local_num_experts: int,
    ep_rank: int = 0,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routing_method_type: int | None = None,
) -> torch.Tensor: ...
