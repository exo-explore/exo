import abc
import torch
from _typeshed import Incomplete
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import get_pp_group as get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.internlm2 import (
    InternLM2Attention as InternLM2Attention,
    InternLM2ForCausalLM as InternLM2ForCausalLM,
    InternLM2MLP as InternLM2MLP,
    InternLM2Model as InternLM2Model,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class InternLM2VEDecoderLayer(nn.Module):
    hidden_size: Incomplete
    attention: Incomplete
    feed_forward: Incomplete
    feed_forward_ve: Incomplete
    attention_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        visual_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class InternLM2VEModel(InternLM2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class InternLM2VEForCausalLM(InternLM2ForCausalLM, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
