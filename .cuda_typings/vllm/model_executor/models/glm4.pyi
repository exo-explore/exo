import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .llama import LlamaModel as LlamaModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Glm4Config as Glm4Config
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
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
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionType as AttentionType

class Glm4Attention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: Glm4Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = ...,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = ...,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Glm4DecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    post_self_attn_layernorm: Incomplete
    post_mlp_layernorm: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Glm4Config | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

ALL_DECODER_LAYER_TYPES: Incomplete

class Glm4Model(LlamaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Glm4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
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
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_spec_layer_idx_from_weight_name(
    config: Glm4Config, weight_name: str
) -> int | None: ...
