import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    get_tp_group as get_tp_group,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    StaticSinkAttention as StaticSinkAttention,
)
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
from vllm.model_executor.layers.mla import (
    MLAModules as MLAModules,
    MultiHeadLatentAttentionWrapper as MultiHeadLatentAttentionWrapper,
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
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts as MixtureOfExperts,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.config import (
    set_default_rope_theta as set_default_rope_theta,
)
from vllm.v1.attention.backend import AttentionType as AttentionType
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend as FlashAttentionDiffKVBackend,
)

def check_ffn_act_fn(act_fn: str): ...

class OpenPanguMLP(nn.Module):
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
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class OpenPanguMoE(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    routed_scaling_factor: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    n_routed_experts: int
    n_shared_experts: int
    is_sequence_parallel: Incomplete
    gate: Incomplete
    enable_eplb: Incomplete
    n_redundant_experts: Incomplete
    n_logical_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class OpenPanguMLAAttention(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    tp_size: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    prefix: Incomplete
    fused_qkv_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    mla_attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class OpenPanguEmbeddedAttention(nn.Module):
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
    attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
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

class OpenPanguSinkAttention(nn.Module):
    hidden_size: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    qk_nope_dim: Incomplete
    qk_rope_dim: Incomplete
    v_channels: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    k_size: Incomplete
    v_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    param_sink_number: Incomplete
    param_sink_with_value: Incomplete
    param_sink_scalar: Incomplete
    param_sink_of_head_num: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    k_layernorm: Incomplete
    attn: Incomplete
    param_sink_key: Incomplete
    param_sink_value: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = ...,
    ) -> None: ...
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor): ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...
    def post_weight_load(self) -> None: ...

class OpenPanguDecoderLayer(nn.Module):
    hidden_size: Incomplete
    layer_idx: Incomplete
    use_mla: Incomplete
    use_sink_attention: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    routed_scaling_factor: Incomplete
    num_hidden_layers: Incomplete
    first_k_dense_replace: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    tp_group: Incomplete
    sandwich_norm: Incomplete
    pre_mlp_layernorm: Incomplete
    post_mlp_layernorm: Incomplete
    def __init__(
        self, config: PretrainedConfig, prefix: str, vllm_config: VllmConfig
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor: ...

class OpenPanguModel(nn.Module):
    fall_back_to_pt_during_load: bool
    config: Incomplete
    num_redundant_experts: Incomplete
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
    def load_attn_mlp_weight(
        self,
        attn_mlp_replace_mapping: list[tuple[str, str, int]],
        params_dict: dict[str, Any],
        weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_params: set[str],
    ) -> bool: ...
    def load_expert_weight(
        self,
        expert_merge_mapping: list[tuple[str, str, int, str]],
        params_dict: dict[str, Any],
        weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_params: set[str],
        flag_dict: dict[str, bool],
    ) -> bool: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def post_weight_load(self) -> None: ...

class OpenPanguModelBase(nn.Module, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    fuse_qkv_a_proj: Incomplete
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

class OpenPanguMoEModel(OpenPanguModelBase, MixtureOfExperts, metaclass=abc.ABCMeta):
    expert_weights: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: int
    moe_layers: Incomplete
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    n_routed_experts: Incomplete
    n_shared_experts: Incomplete
    num_redundant_experts: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...

class OpenPanguEmbeddedModel(OpenPanguModelBase, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class PanguEmbeddedForCausalLM(OpenPanguEmbeddedModel, metaclass=abc.ABCMeta): ...
class PanguUltraMoEForCausalLM(OpenPanguMoEModel, metaclass=abc.ABCMeta): ...
class PanguProMoEV2ForCausalLM(OpenPanguMoEModel, metaclass=abc.ABCMeta): ...
