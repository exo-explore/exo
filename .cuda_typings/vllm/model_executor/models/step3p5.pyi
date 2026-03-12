import abc
import torch
from .interfaces import MixtureOfExperts as MixtureOfExperts, SupportsPP as SupportsPP
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    WeightsMapper as WeightsMapper,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from torch.nn.parameter import Parameter
from typing import Any
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_dp_group as get_dp_group,
    get_ep_group as get_ep_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    get_tp_group as get_tp_group,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import (
    SiluAndMul as SiluAndMul,
    SwigluStepAndMul as SwigluStepAndMul,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
    SharedFusedMoE as SharedFusedMoE,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as GemmaRMSNorm
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
from vllm.model_executor.layers.quantization.base_config import (
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
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionType as AttentionType

logger: Incomplete

class FP32ReplicatedLinear(ReplicatedLinear):
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]: ...

class Step3p5MLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    prefix: Incomplete
    hidden_size: Incomplete
    limit: Incomplete
    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3p5Attention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    layer_idx: Incomplete
    rank: Incomplete
    partial_rotary_factor: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    rope_theta: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    use_head_wise_attn_gate: Incomplete
    g_proj: Incomplete
    use_rope: bool
    attn: Incomplete
    max_position_embeddings: Incomplete
    rotary_dim: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = ...,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float | list[float] | None = 10000,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rope_scaling: dict[str, Any] | None = None,
        prefix: str = "",
        attn_type: str = ...,
        sliding_window: int | None = None,
        use_head_wise_attn_gate: bool = False,
        layer_types: list = None,
        use_rope_layers: list = None,
        yarn_only_types: list = None,
        swa_num_attention_heads: int | None = None,
        partial_rotary_factor: float = 1.0,
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class FusedMoEBlock(nn.Module):
    tp_size: Incomplete
    layer_idx: Incomplete
    ep_size: Incomplete
    ep_rank: Incomplete
    hidden_size: Incomplete
    enable_eplb: Incomplete
    n_routed_experts: Incomplete
    n_logical_experts: Incomplete
    n_redundant_experts: Incomplete
    n_physical_experts: Incomplete
    n_local_physical_experts: Incomplete
    physical_expert_start: Incomplete
    physical_expert_end: Incomplete
    gate: Incomplete
    use_moe_router_bias: Incomplete
    routed_scaling_factor: Incomplete
    router_bias: Incomplete
    need_fp32_gate: Incomplete
    share_expert: Incomplete
    experts: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Step3p5DecoderLayer(nn.Module):
    hidden_size: Incomplete
    layer_idx: Incomplete
    self_attn: Incomplete
    use_moe: bool
    tp_group: Incomplete
    use_fused_all_reduce: Incomplete
    moe: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    prefix: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def add_and_maybe_inplace_all_reduce(
        self, in1: torch.Tensor, in2: torch.Tensor
    ) -> torch.Tensor: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Step3p5Model(nn.Module):
    vllm_config: Incomplete
    vocab_size: Incomplete
    config: Incomplete
    moe_num_experts: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Step3p5ForCausalLM(
    nn.Module, SupportsPP, MixtureOfExperts, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    moe_layers: list[FusedMoEBlock]
    expert_weights: Incomplete
    num_moe_layers: Incomplete
    num_expert_groups: int
    num_shared_experts: int
    num_logical_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_routed_experts: Incomplete
    num_redundant_experts: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ): ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def get_spec_layer_idx_from_weight_name(
    config: ModelConfig, weight_name: str
) -> int | None: ...
