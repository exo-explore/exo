import torch
from _typeshed import Incomplete
from torch import nn as nn
from vllm.model_executor.layers.fused_moe.fused_moe import (
    zero_experts_compute_triton as zero_experts_compute_triton,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as FusedMoE

class ZeroExpertFusedMoE(FusedMoE):
    zero_expert_num: int
    zero_expert_type: Incomplete
    custom_routing_function: Incomplete
    def __init__(
        self, zero_expert_num: int, zero_expert_type: str, router: nn.Module, **kwargs
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor: ...
