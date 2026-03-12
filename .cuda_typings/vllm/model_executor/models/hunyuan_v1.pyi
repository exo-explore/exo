import abc
import torch
from .interfaces import (
    MixtureOfExperts as MixtureOfExperts,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
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
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionType as AttentionType

class HunYuanMLP(nn.Module):
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
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None: ...
    def forward(self, x): ...

class HunYuanAttention(nn.Module):
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
    use_qk_norm: Incomplete
    layer_id: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    query_layernorm: Incomplete
    key_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor: ...

class HunYuanCrossAttention(nn.Module):
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
    use_qk_norm: Incomplete
    layer_id: Incomplete
    q_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    query_layernorm: Incomplete
    key_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor: ...

class HunYuanSparseMoeBlock(nn.Module):
    tp_size: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    n_routed_experts: Incomplete
    enable_eplb: Incomplete
    n_logical_experts: Incomplete
    n_redundant_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    gate: Incomplete
    shared_mlp: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = -1,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class HunYuanDecoderLayer(nn.Module):
    layer_id: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_id: int = -1,
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_states: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class HunYuanModel(nn.Module):
    num_redundant_experts: Incomplete
    config: Incomplete
    quant_config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    aux_hidden_state_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...

class HunyuanV1ModelBase(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class HunYuanMoEV1Base(HunyuanV1ModelBase, MixtureOfExperts, metaclass=abc.ABCMeta):
    expert_weights: Incomplete
    num_expert_groups: int
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_routed_experts: Incomplete
    num_redundant_experts: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

class HunYuanDenseV1Base(HunyuanV1ModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class HunYuanDenseV1ForCausalLM(HunYuanDenseV1Base): ...
class HunYuanMoEV1ForCausalLM(HunYuanMoEV1Base, metaclass=abc.ABCMeta): ...
