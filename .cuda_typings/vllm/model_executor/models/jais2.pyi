import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
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
from transformers import Jais2Config as Jais2Config
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import (
    ReLUSquaredActivation as ReLUSquaredActivation,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
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
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class Jais2MLP(nn.Module):
    up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class Jais2Attention(nn.Module):
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
        config: Jais2Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Jais2DecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, config: Jais2Config, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_quant_config(
        self, vllm_config: VllmConfig
    ) -> QuantizationConfig | None: ...

class Jais2Model(nn.Module):
    config: Incomplete
    quant_config: Incomplete
    vocab_size: Incomplete
    org_vocab_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ...,
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Jais2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    config: Incomplete
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
