import abc
import torch
from .interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Cohere2Config as Cohere2Config, CohereConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
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
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
    row_parallel_weight_loader as row_parallel_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors

def layer_norm_func(hidden_states, weight, variance_epsilon): ...

class LayerNorm(nn.Module):
    weight: Incomplete
    variance_epsilon: Incomplete
    def __init__(self, param_shape=None, eps: float = 1e-05) -> None: ...
    def forward(self, hidden_states, residuals=None): ...

class CohereMLP(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: CohereConfig | Cohere2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class CohereAttention(nn.Module):
    config: Incomplete
    attention_dropout: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    use_qk_norm: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    v1: Incomplete
    sliding_window: Incomplete
    attn: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(
        self,
        config: CohereConfig | Cohere2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class CohereDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    def __init__(
        self,
        config: CohereConfig | Cohere2Config,
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

class CohereModel(nn.Module):
    quant_config: Incomplete
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class CohereForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    config: Incomplete
    quant_config: Incomplete
    logits_processor: Incomplete
    model: Incomplete
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
