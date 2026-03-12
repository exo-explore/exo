import abc
import torch
from .interfaces import (
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import (
    FatreluAndMul as FatreluAndMul,
    SiluAndMul as SiluAndMul,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import (
    fused_experts as fused_experts,
    fused_topk as fused_topk,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

class MiniCPMMoE(nn.Module):
    tp_size: Incomplete
    num_total_experts: Incomplete
    top_k: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    params_dtype: Incomplete
    gate: Incomplete
    ws: Incomplete
    w2s: Incomplete
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        tp_size: int | None = None,
        prefix: str = "",
    ) -> None: ...
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ): ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MiniCPMMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_act_param: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class MiniCPMAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class MiniCPMDecoderLayer(nn.Module):
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    hidden_size: Incomplete
    max_position_embeddings: Incomplete
    prefix: Incomplete
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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class MiniCPMModel(nn.Module):
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    num_experts: Incomplete
    norm: Incomplete
    aux_hidden_state_layers: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class MiniCPMForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    prefix: Incomplete
    vllm_config: Incomplete
    config: Incomplete
    cache_config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    scale_width: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
