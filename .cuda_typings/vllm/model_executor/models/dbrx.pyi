import abc
import torch
import torch.nn as nn
from .interfaces import SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import DbrxConfig as DbrxConfig
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import (
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
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class DbrxRouter(nn.Module):
    tp_size: Incomplete
    num_total_experts: Incomplete
    d_model: Incomplete
    layer: Incomplete
    def __init__(
        self, config: DbrxConfig, params_dtype: torch.dtype | None = None
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class DbrxExperts(FusedMoE):
    config: Incomplete
    d_model: Incomplete
    intermediate_size: Incomplete
    def __init__(
        self,
        config: DbrxConfig,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None: ...
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        param_name: str,
    ): ...

class DbrxMoE(nn.Module):
    d_model: Incomplete
    params_dtype: Incomplete
    router: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: DbrxConfig,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class DbrxAttention(nn.Module):
    d_model: Incomplete
    total_num_heads: Incomplete
    head_dim: Incomplete
    total_num_kv_heads: Incomplete
    clip_qkv: Incomplete
    max_position: Incomplete
    Wqkv: Incomplete
    out_proj: Incomplete
    rotary_emb: Incomplete
    tp_size: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, position_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class DbrxFusedNormAttention(nn.Module):
    d_model: Incomplete
    attn: Incomplete
    norm_1: Incomplete
    norm_2: Incomplete
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, position_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class DbrxBlock(nn.Module):
    norm_attn_norm: Incomplete
    ffn: Incomplete
    def __init__(
        self,
        config: DbrxConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, position_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class DbrxModel(nn.Module):
    quant_config: Incomplete
    wte: Incomplete
    norm_f: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class DbrxForCausalLM(nn.Module, SupportsPP, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    transformer: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
