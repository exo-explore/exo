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
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from typing import Any
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
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
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
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

logger: Incomplete

class Qwen3MoeMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    expert_gate: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class Qwen3MoeSparseMoeBlock(nn.Module):
    tp_size: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    n_routed_experts: Incomplete
    is_sequence_parallel: Incomplete
    enable_eplb: Incomplete
    n_logical_experts: Incomplete
    n_redundant_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    gate: Incomplete
    shared_expert_gate: Incomplete
    shared_expert: Incomplete
    experts: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3MoeAttention(nn.Module):
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
    dual_chunk_attention_config: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Qwen3MoeDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Qwen3MoeModel(nn.Module):
    num_redundant_experts: Incomplete
    vocab_size: Incomplete
    config: Incomplete
    quant_config: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    aux_hidden_state_layers: tuple[int, ...]
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[torch.nn.Module] = ...,
    ) -> None: ...
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
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3MoeForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle3,
    MixtureOfExperts,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    embedding_modules: Incomplete
    fall_back_to_pt_during_load: bool
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    expert_weights: Incomplete
    moe_layers: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: int
    num_shared_experts: int
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_routed_experts: Incomplete
    num_redundant_experts: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
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
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
