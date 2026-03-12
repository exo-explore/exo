import abc
import torch
from _typeshed import Incomplete
from torch import nn
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.models.olmoe import (
    OlmoeAttention as OlmoeAttention,
    OlmoeForCausalLM as OlmoeForCausalLM,
)
from vllm.transformers_utils.configs import FlexOlmoConfig as FlexOlmoConfig

logger: Incomplete

class FlexOlmoAttention(OlmoeAttention):
    k_norm: Incomplete
    q_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class FlexOlmoMoE(nn.Module):
    gate: Incomplete
    experts: Incomplete
    top_k: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class FlexOlmoDecoderLayer(nn.Module):
    self_attn: Incomplete
    post_attention_layernorm: Incomplete
    post_feedforward_layernorm: Incomplete
    mlp: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class FlexOlmoForCausalLM(OlmoeForCausalLM, metaclass=abc.ABCMeta):
    fall_back_to_pt_during_load: bool
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ...,
    ) -> None: ...
