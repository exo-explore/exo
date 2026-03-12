import abc
import torch
from .llama import (
    LlamaForCausalLM as LlamaForCausalLM,
    LlamaMLP as LlamaMLP,
    LlamaModel as LlamaModel,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    extract_layer_index as extract_layer_index,
    fast_topk as fast_topk,
    is_pp_missing_parameter as is_pp_missing_parameter,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import Llama4TextConfig as Llama4TextConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    ChunkedLocalAttention as ChunkedLocalAttention,
)
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts as MixtureOfExperts
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as sequence_parallel_chunk,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer

logger: Incomplete

class Llama4MoE(nn.Module):
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    tp_size: Incomplete
    top_k: Incomplete
    is_sequence_parallel: Incomplete
    ep_group: Incomplete
    ep_rank: Incomplete
    ep_size: Incomplete
    router: Incomplete
    shared_expert: Incomplete
    enable_eplb: Incomplete
    n_redundant_experts: Incomplete
    n_routed_experts: int
    n_logical_experts: Incomplete
    n_shared_experts: int
    n_local_experts: int
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    experts: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states): ...

class Llama4Attention(nn.Module):
    layer_idx: Incomplete
    hidden_size: Incomplete
    no_rope_layers: Incomplete
    nope: Incomplete
    use_qk_norm: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    attn_temperature_tuning: Incomplete
    floor_scale: Incomplete
    attn_scale: Incomplete
    max_position_embeddings: Incomplete
    n_rep: Incomplete
    qk_norm: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Llama4DecoderLayer(nn.Module):
    layer_idx: Incomplete
    global_layer: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    feed_forward: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Llama4TextConfig | None = None,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Llama4Model(LlamaModel):
    num_experts: Incomplete
    n_redundant_experts: Incomplete
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[Llama4DecoderLayer] = ...,
    ) -> None: ...
    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        loaded_params: set[str],
        expert_params_mapping: list[tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Llama4ForCausalLM(LlamaForCausalLM, MixtureOfExperts, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    expert_weights: Incomplete
    moe_layers: Incomplete
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int
    def set_moe_parameters(self) -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def permute_qk_weight_for_rotary(
        self, name: str, loaded_weight: torch.Tensor
    ) -> tuple[str, torch.Tensor]: ...
