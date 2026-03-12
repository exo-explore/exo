import abc
import torch
import torch.nn as nn
from .interfaces import (
    HasInnerState as HasInnerState,
    IsHybrid as IsHybrid,
    MixtureOfExperts as MixtureOfExperts,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
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
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group as get_ep_group,
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
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
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.short_conv import ShortConv as ShortConv
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
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import Lfm2MoeConfig as Lfm2MoeConfig

class Lfm2MoeMlp(nn.Module):
    w13: Incomplete
    w2: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Lfm2MoeSparseMoeBlock(nn.Module):
    tp_size: Incomplete
    routed_scaling_factor: Incomplete
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
    experts: Incomplete
    def __init__(
        self,
        config: Lfm2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Lfm2MoeAttention(nn.Module):
    layer_idx: Incomplete
    hidden_size: Incomplete
    num_kv_heads: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    q_layernorm: Incomplete
    k_layernorm: Incomplete
    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Lfm2MoeAttentionDecoderLayer(nn.Module):
    prefix: Incomplete
    config: Incomplete
    layer_idx: Incomplete
    self_attn: Incomplete
    feed_forward: Incomplete
    operator_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Lfm2MoeShortConvDecoderLayer(nn.Module):
    layer_idx: Incomplete
    short_conv: Incomplete
    feed_forward: Incomplete
    operator_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(
        self,
        config: Lfm2MoeConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None, **kwargs
    ): ...

class Lfm2MoeModel(nn.Module):
    num_redundant_experts: Incomplete
    config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    make_empty_intermediate_tensors: Incomplete
    embedding_norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Lfm2MoeForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    MixtureOfExperts,
    metaclass=abc.ABCMeta,
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    embedding_modules: Incomplete
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]: ...
    config: Incomplete
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
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
