import abc
import torch
from .ernie45_moe import Ernie4_5_MoeMLP as Ernie4_5_MoeMLP
from .interfaces import SupportsPP as SupportsPP
from .utils import (
    PPMissingLayer as PPMissingLayer,
    extract_layer_index as extract_layer_index,
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
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
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
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding as Ernie4_5_VLRotaryEmbedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.config import (
    set_default_rope_theta as set_default_rope_theta,
)

logger: Incomplete

class Ernie4_5_VLMoeMLP(Ernie4_5_MoeMLP):
    shared_experts: Incomplete
    def __init__(
        self, shared_experts: torch.nn.Module | None = None, **kwargs
    ) -> None: ...
    def forward(self, x): ...

class Ernie4_5_VLMoeAttention(nn.Module):
    layer_idx: Incomplete
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
        rope_parameters: dict[str, Any],
        head_dim: int | None = None,
        freq_allocation: int = 20,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Ernie4_5_VLMoeMoE(nn.Module):
    layer_idx: Incomplete
    tp_size: Incomplete
    has_shared_experts: Incomplete
    hidden_size: Incomplete
    e_score_correction_bias: Incomplete
    shared_experts: Incomplete
    text_experts_gate: Incomplete
    text_experts: Incomplete
    vision_experts_gate: Incomplete
    vision_experts: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor: ...

class Ernie4_5_VLMoeDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    layer_idx: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
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
        visual_token_mask: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor: ...

class Ernie4_5_VLMoeModel(nn.Module):
    vocab_size: Incomplete
    config: Incomplete
    im_patch_id: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...

class Ernie4_5_VLMoeForCausalLM(nn.Module, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    fall_back_to_pt_during_load: bool
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
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
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
