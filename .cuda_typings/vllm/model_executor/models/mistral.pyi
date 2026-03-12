import abc
import torch
from .utils import AutoWeightsLoader as AutoWeightsLoader
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import LlamaConfig as LlamaConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.llama import (
    LlamaAttention as LlamaAttention,
    LlamaDecoderLayer as LlamaDecoderLayer,
    LlamaForCausalLM as LlamaForCausalLM,
    LlamaModel as LlamaModel,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionType as AttentionType

class MistralMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        gate_up_proj_bias: bool | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def forward(self, x): ...

class MistralAttention(LlamaAttention):
    do_llama_4_scaling: Incomplete
    llama_4_scaling_original_max_position_embeddings: Incomplete
    llama_4_scaling_beta: Incomplete
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = ...,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class MistralDecoderLayer(LlamaDecoderLayer):
    layer_idx: Incomplete
    ada_rms_norm_t_cond: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        t_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class MistralModel(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ...,
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        t_cond: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]
    ): ...

class MistralForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    embedding_modules: dict[str, str]
    mistral_mapping: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ...,
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def maybe_remap_mistral(
        self, name: str, loaded_weight: torch.Tensor
    ) -> tuple[str, torch.Tensor]: ...
