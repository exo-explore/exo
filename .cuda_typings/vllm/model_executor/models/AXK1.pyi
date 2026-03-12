import abc
import torch
from .interfaces import (
    MixtureOfExperts as MixtureOfExperts,
    SupportsEagle as SupportsEagle,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
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
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
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
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekAttention as DeepseekAttention,
    DeepseekV2MLP as DeepseekV2MLP,
    yarn_get_mscale as yarn_get_mscale,
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.platforms import current_platform as current_platform
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.AXK1 import AXK1Config as AXK1Config

logger: Incomplete

class AXK1MLP(DeepseekV2MLP): ...

class AXK1MoE(nn.Module):
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
    is_rocm_aiter_moe_enabled: Incomplete
    is_fusion_moe_shared_experts_enabled: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config: AXK1Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class AXK1Attention(nn.Module):
    hidden_size: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    num_heads: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    q_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: AXK1Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor: ...

class AXK1MLAAttention(nn.Module):
    hidden_size: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    num_heads: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    fused_qkv_a_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    mla_attn: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: AXK1Config,
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
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor: ...

class AXK1DecoderLayer(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    layer_idx: Incomplete
    use_mha: Incomplete
    self_attn: Incomplete
    is_layer_sparse: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    post_mlp_layernorm: Incomplete
    routed_scaling_factor: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, prefix: str, config: AXK1Config | None = None
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class AXK1Model(nn.Module):
    fall_back_to_pt_during_load: bool
    config: Incomplete
    device: Incomplete
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

class AXK1MixtureOfExperts(MixtureOfExperts, metaclass=abc.ABCMeta):
    moe_mlp_layers: list[AXK1MoE]
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int
    def extract_moe_parameters(self, example_moe: AXK1MoE | None): ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...

class AXK1ForCausalLM(
    nn.Module,
    SupportsPP,
    AXK1MixtureOfExperts,
    SupportsLoRA,
    SupportsEagle,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    model_cls = AXK1Model
    config: Incomplete
    quant_config: Incomplete
    use_mha: Incomplete
    fuse_qkv_a_proj: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    num_moe_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    expert_weights: Incomplete
    num_expert_groups: Incomplete
    moe_layers: Incomplete
    moe_mlp_layers: Incomplete
    def set_moe_parameters(self) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_spec_layer_idx_from_weight_name(
    config: AXK1Config, weight_name: str
) -> int | None: ...
