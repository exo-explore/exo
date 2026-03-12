import abc
import torch
from .interfaces import SupportsPP as SupportsPP, SupportsQuant as SupportsQuant
from .utils import (
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
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
from vllm.transformers_utils.configs.arctic import ArcticConfig as ArcticConfig

logger: Incomplete

class ArcticMLP(nn.Module):
    hidden_size: Incomplete
    expert_id: Incomplete
    ffn_dim: Incomplete
    w13: Incomplete
    w2: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        config: ArcticConfig,
        expert_id: int = -1,
        is_residual_mlp: bool = False,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states): ...

class ArcticMoE(nn.Module):
    tp_size: Incomplete
    hidden_size: Incomplete
    num_experts: Incomplete
    layer_id: Incomplete
    top_k: Incomplete
    intermediate_size: Incomplete
    is_moe_layer: Incomplete
    reduce_results: Incomplete
    params_dtype: Incomplete
    mlp: Incomplete
    gate: Incomplete
    ws: Incomplete
    w2s: Incomplete
    def __init__(
        self,
        config: ArcticConfig,
        tp_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ): ...
    def local_moe_fused(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def forward(self, hidden_states: torch.Tensor): ...

class ArcticAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    max_position_embeddings: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: ArcticConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class ArcticDecoderLayer(nn.Module):
    hidden_size: Incomplete
    use_residual: Incomplete
    self_attn: Incomplete
    block_sparse_moe: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    residual_layernorm: Incomplete
    residual_mlp: Incomplete
    def __init__(
        self,
        config: ArcticConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class ArcticModel(nn.Module):
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

class ArcticForCausalLM(nn.Module, SupportsPP, SupportsQuant, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    model: Incomplete
    vocab_size: Incomplete
    lm_head: Incomplete
    num_experts: Incomplete
    num_experts_per_tok: Incomplete
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
